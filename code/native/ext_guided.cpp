// NATIVE PLANNER CHECK for guidance-weighted lifelong MAPF.
//
// Every per-step decision is made by the UPSTREAM, COMPILED PIBT of Kei18/lacam3 (commit
// 55347c8): `PIBT::set_new_config` -- its candidate ranking, its swap detector, its random
// tie-breaker.  Nothing from this repository's `code/mapf/pibt.py` or `code/mapf/sim.py` runs
// here.  This file supplies only the lifelong protocol around that planner:
//
//   * native task generation: its own std::mt19937, uniform distinct starts over the largest
//     connected component, and either i.i.d. uniform goals over that component or alternating
//     pickup/station legs when a pool file is given;
//   * elapsed-time priorities with a fixed per-agent tie-breaker;
//   * the guidance field, injected the only way a native PIBT can consume it -- as the
//     distance-to-goal table it ranks candidates with.
//
// How guidance enters, and why this is the faithful translation.  lacam3's PIBT scores a
// candidate u as `D->get(i,u) + tie_breaker(u)`, with an INTEGER table and a tie-breaker drawn
// uniformly from [0,1).  A unit difference in the table therefore always dominates the
// tie-breaker, and equal table entries are broken at random.  This driver fills that table with
//
//     table[i][u] = round( dist_w(u -> goal_i) / eps_tie )
//
// where dist_w is the weighted shortest-path distance in the DIRECTED guidance-weighted graph
// (arc cost = the guidance weight of that directed arc), computed by Dijkstra on the reverse
// graph, exactly as `mapf.pibt.dist_table` does.  The rounding matches `mapf.pibt.PIBT._pibt`,
// which sorts candidates on `round(D[u] / eps_tie)`.  An earlier build of this driver used
// `floor` here; that is a different bin partition, so both matrices were re-run and the floor
// runs are retained under `results/native_check/runs_floor/` and `runs_bench_floor/`.
// Quantising by eps_tie is what the Python
// implementation's `eps_tie` bin comparison does, and with eps_tie = 1 (the value used
// throughout the paper) it is *literally* lacam3's own integer semantics.  So the native arm
// tests the guidance field and the tie rule together, on third-party planner code.
//
// What is deliberately NOT reproduced: the Python planner's `tau` stall escape (an agent blocked
// more than tau steps demotes "wait"), which upstream PIBT has no equivalent of, and the Python
// swap detector's own emulation caps.  Those differences are the point of a native check and are
// reported, not hidden.
//
// Build (from this directory):
//   c++ -std=c++17 -O3 -I<envelope>/_probes/ext -o ext_guided ext_guided.cpp \
//       <envelope>/_probes/ext/lacam3/build/lacam3/liblacam3.a -lpthread
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "lacam3/lacam3/include/dist_table.hpp"
#include "lacam3/lacam3/include/graph.hpp"
#include "lacam3/lacam3/include/instance.hpp"
#include "lacam3/lacam3/include/pibt.hpp"

static const int UNREACH = 1 << 28;

int main(int argc, char *argv[])
{
  std::string map_file, w_file, pool_file;
  int N = 0, steps = 1000, warmup = 0, seed = 0, bin_steps = 100, swap_on = 1;
  double eps_tie = 1.0;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto nxt = [&]() { return std::string(argv[++i]); };
    if (a == "-m" || a == "--map") map_file = nxt();
    else if (a == "-N") N = std::stoi(nxt());
    else if (a == "--steps") steps = std::stoi(nxt());
    else if (a == "--warmup") warmup = std::stoi(nxt());
    else if (a == "--seed") seed = std::stoi(nxt());
    else if (a == "--bin") bin_steps = std::stoi(nxt());
    else if (a == "--weights") w_file = nxt();
    else if (a == "--pools") pool_file = nxt();
    else if (a == "--eps-tie") eps_tie = std::stod(nxt());
    else if (a == "--no-swap") swap_on = 0;
  }
  if (map_file.empty() || N <= 0) { fprintf(stderr, "need -m MAP -N NUM\n"); return 2; }

  auto G = new Graph(map_file);
  const int V = (int)G->V.size();
  const int Wd = G->width;

  // grid index (y*width + x) -> lacam3 vertex id
  auto vid_of_grid = [&](long long gidx) -> int {
    if (gidx < 0 || gidx >= (long long)G->U.size() || G->U[gidx] == nullptr) return -1;
    return G->U[gidx]->id;
  };

  // ---- guidance weights on directed arcs, indexed by lacam3 vertex ids -------------------
  // wadj[u] holds (v, cost) for every out-arc u->v; default cost 1 on all grid arcs.
  std::vector<std::vector<std::pair<int, double>>> wadj(V);
  for (int u = 0; u < V; ++u)
    for (auto &m : G->V[u]->neighbor) wadj[u].emplace_back(m->id, 1.0);
  long long n_arcs_read = 0;
  if (!w_file.empty()) {
    std::ifstream fh(w_file);
    if (!fh) { fprintf(stderr, "cannot open weights %s\n", w_file.c_str()); return 2; }
    long long E = 0; fh >> E;
    for (long long e = 0; e < E; ++e) {
      long long gu, gv; double wt;
      fh >> gu >> gv >> wt;
      const int u = vid_of_grid(gu), v = vid_of_grid(gv);
      if (u < 0 || v < 0) { fprintf(stderr, "weight arc off the free set\n"); return 2; }
      bool hit = false;
      for (auto &pr : wadj[u]) if (pr.first == v) { pr.second = wt; hit = true; break; }
      if (!hit) { fprintf(stderr, "weight arc %lld->%lld is not a grid edge\n", gu, gv); return 2; }
      ++n_arcs_read;
    }
    long long n_grid_arcs = 0;
    for (int u = 0; u < V; ++u) n_grid_arcs += (long long)wadj[u].size();
    if (n_arcs_read != n_grid_arcs) {
      fprintf(stderr, "arc count mismatch: file %lld, map %lld\n", n_arcs_read, n_grid_arcs);
      return 2;
    }
  }
  // reverse adjacency, for a Dijkstra rooted at the goal
  std::vector<std::vector<std::pair<int, double>>> radj(V);
  for (int u = 0; u < V; ++u)
    for (auto &pr : wadj[u]) radj[pr.first].emplace_back(u, pr.second);

  // ---- largest connected component (starts and goals are drawn from it only) -------------
  std::vector<int> comp(V, -1), best_comp;
  for (int s0 = 0; s0 < V; ++s0) {
    if (comp[s0] != -1) continue;
    std::vector<int> cur;
    std::queue<Vertex *> Q;
    comp[s0] = s0; Q.push(G->V[s0]);
    while (!Q.empty()) {
      auto n = Q.front(); Q.pop();
      cur.push_back(n->id);
      for (auto &m : n->neighbor) if (comp[m->id] == -1) { comp[m->id] = s0; Q.push(m); }
    }
    if (cur.size() > best_comp.size()) best_comp = std::move(cur);
  }
  const int Vc = (int)best_comp.size();
  if (N > Vc) { fprintf(stderr, "N > |largest component|\n"); return 2; }
  std::vector<char> in_comp(V, 0);
  for (int v : best_comp) in_comp[v] = 1;

  // ---- endpoint pools (warehouse task model) --------------------------------------------
  std::vector<int> picks, stations;
  const bool warehouse = !pool_file.empty();
  if (warehouse) {
    std::ifstream fh(pool_file);
    if (!fh) { fprintf(stderr, "cannot open pools %s\n", pool_file.c_str()); return 2; }
    long long np = 0, ns = 0; fh >> np >> ns;
    auto rd = [&](long long k, std::vector<int> &dst) {
      for (long long j = 0; j < k; ++j) {
        long long g; fh >> g;
        const int v = vid_of_grid(g);
        if (v < 0) { fprintf(stderr, "pool cell off the free set\n"); exit(2); }
        if (in_comp[v]) dst.push_back(v);
      }
    };
    rd(np, picks); rd(ns, stations);
    if (picks.empty() || stations.empty()) { fprintf(stderr, "empty pool\n"); return 2; }
  }

  // ---- quantised weighted distance rows, cached per goal ---------------------------------
  std::unordered_map<int, std::vector<int>> cache;
  auto dist_to = [&](int gv) -> const std::vector<int> & {
    auto it = cache.find(gv);
    if (it != cache.end()) return it->second;
    std::vector<double> d(V, std::numeric_limits<double>::infinity());
    using QE = std::pair<double, int>;
    std::priority_queue<QE, std::vector<QE>, std::greater<QE>> PQ;
    d[gv] = 0.0; PQ.emplace(0.0, gv);
    while (!PQ.empty()) {
      auto [dv, v] = PQ.top(); PQ.pop();
      if (dv > d[v]) continue;
      for (auto &pr : radj[v]) {                 // reverse arc: cost of pr.first -> v
        const double nd = dv + pr.second;
        if (nd < d[pr.first]) { d[pr.first] = nd; PQ.emplace(nd, pr.first); }
      }
    }
    std::vector<int> q(V, UNREACH);
    for (int v = 0; v < V; ++v)
      if (std::isfinite(d[v])) q[v] = (int)std::llround(d[v] / eps_tie);
    return cache.emplace(gv, std::move(q)).first->second;
  };

  // ---- native task generation ------------------------------------------------------------
  std::mt19937 MT(seed);
  std::vector<int> perm = best_comp;
  std::shuffle(perm.begin(), perm.end(), MT);
  std::uniform_int_distribution<int> pick_free(0, Vc - 1);
  std::vector<int> phase(N, 0);
  auto draw_goal = [&](int i, int avoid) {
    if (!warehouse) {
      int g = best_comp[pick_free(MT)];
      for (int k = 0; k < 19 && g == avoid; ++k) g = best_comp[pick_free(MT)];
      return g;
    }
    const std::vector<int> &pool = (phase[i] == 0) ? picks : stations;
    phase[i] = 1 - phase[i];
    std::uniform_int_distribution<int> pk(0, (int)pool.size() - 1);
    int g = pool[pk(MT)];
    if (g == avoid) g = pool[pk(MT)];
    return g;
  };

  Config starts(N), goals(N);
  std::vector<int> gid(N);
  for (int i = 0; i < N; ++i) {
    starts[i] = G->V[perm[i]];
    gid[i] = draw_goal(i, perm[i]);
    goals[i] = G->V[gid[i]];
  }

  auto ins = new Instance(G, starts, goals, (uint)N);
  ins->delete_graph_after_used = false;
  auto D = new DistTable(ins);
  for (int i = 0; i < N; ++i) D->table[i] = dist_to(gid[i]);
  auto pibt = new PIBT(ins, D, seed, swap_on != 0, nullptr);

  std::vector<double> elapsed(N, 0.0), tie(N);
  std::uniform_real_distribution<double> U(0.0, 1.0);
  for (int i = 0; i < N; ++i) tie[i] = U(MT);

  Config Q_from = starts, Q_to(N, nullptr);
  std::vector<int> order(N);
  const int nb = std::max(1, steps / bin_steps);
  std::vector<long long> bdone(nb, 0), bdelay(nb, 0), bwait(nb, 0);
  long long fails = 0;

  for (int t = 0; t < steps; ++t) {
    for (int i = 0; i < N; ++i) { order[i] = i; Q_to[i] = nullptr; }
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      return elapsed[a] + tie[a] > elapsed[b] + tie[b];
    });
    if (!pibt->set_new_config(Q_from, Q_to, order)) {
      ++fails;
      for (int i = 0; i < N; ++i) if (Q_to[i] == nullptr) Q_to[i] = Q_from[i];
    }
    int b = t / bin_steps; if (b >= nb) b = nb - 1;
    for (int i = 0; i < N; ++i) {
      if (Q_to[i]->id == Q_from[i]->id) ++bwait[b];
      Q_from[i] = Q_to[i];
      if (Q_from[i]->id == gid[i]) {
        ++bdone[b];
        bdelay[b] += (long long)elapsed[i] + 1;
        elapsed[i] = 0.0;
        gid[i] = draw_goal(i, Q_from[i]->id);
        ins->goals[i] = G->V[gid[i]];
        D->table[i] = dist_to(gid[i]);
      } else {
        elapsed[i] += 1.0;
      }
    }
  }

  const int wb = warmup / bin_steps;
  long long done = 0, delay = 0, wait = 0;
  for (int b = wb; b < nb; ++b) { done += bdone[b]; delay += bdelay[b]; wait += bwait[b]; }
  const long long msteps = (long long)(nb - wb) * bin_steps;
  printf("{\"map\": \"%s\", \"N\": %d, \"steps\": %d, \"warmup\": %d, \"seed\": %d, "
         "\"swap\": %d, \"eps_tie\": %g, \"weights\": \"%s\", \"pools\": \"%s\", "
         "\"V\": %d, \"V_component\": %d, \"width\": %d, \"n_arcs_read\": %lld, "
         "\"pibt_fail_steps\": %lld, \"completed\": %lld, \"measured_steps\": %lld, "
         "\"throughput\": %.6f, \"mean_task_delay\": %.6f, \"wait_fraction\": %.6f, \"bins\": [",
         map_file.c_str(), N, steps, warmup, seed, swap_on, eps_tie, w_file.c_str(),
         pool_file.c_str(), V, Vc, Wd, n_arcs_read, fails, done, msteps,
         (double)done / (double)msteps,
         done ? (double)delay / (double)done : 0.0,
         (double)wait / (double)(msteps * N));
  for (int b = 0; b < nb; ++b) printf("%s%lld", b ? ", " : "", bdone[b]);
  printf("]}\n");

  delete pibt; delete D; delete ins; delete G;
  return 0;
}

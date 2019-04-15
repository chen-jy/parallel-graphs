#include <algorithm>
#include <cstdio>
#include <list>
#include <omp.h>
#include <queue>
#include <string>
#include <vector>
#include "getopt.h"

#define INF 0x3f3f3f3f
#define NUM_ITERATIONS 10

using namespace std;

typedef vector<vector<pair<int, double>>> AdjList;

// MST sequential baseline
vector<int> prim_seq(AdjList graph) {
	vector<double> dist(graph.size(), INF);
	vector<bool> vis(graph.size(), 0);
	vector<int> pred(graph.size(), -1);
	priority_queue<pair<double, int>> q;

	dist[0] = 0;
	q.push({ 0, 0 });

	while (!q.empty()) {
		int u = q.top().second;
		q.pop();

		if (vis[u]) continue;
		vis[u] = 1;

		for (int i = 0; i < graph[u].size(); i++) {
			int v = graph[u][i].first;
			double w = graph[u][i].second;

			if (!vis[v] && w < dist[v]) {
				dist[v] = w, pred[v] = u;
				q.push({ -w, v });
			}
		}
	}

	return pred;
}

// MST parallel implementation 1
vector<int> prim_par(AdjList graph) {
	vector<double> dist(graph.size(), INF);
	vector<bool> vis(graph.size(), 0);
	vector<int> pred(graph.size(), -1);
	priority_queue<pair<double, int>> q;

	omp_lock_t qlock;
	omp_init_lock(&qlock);

	dist[0] = 0;
	q.push({ 0, 0 });

	while (!q.empty()) {
		int u = q.top().second;
		q.pop();

		if (vis[u]) continue;
		vis[u] = 1;

		#pragma omp parallel for default(none) shared(dist, vis, pred, q, qlock, u)
		for (int i = 0; i < graph[u].size(); i++) {
			int v = graph[u][i].first;
			double w = graph[u][i].second;

			if (!vis[v] && w < dist[v]) {
				dist[v] = w, pred[v] = u;
				omp_set_lock(&qlock);
				q.push({ -w, v });
				omp_unset_lock(&qlock);
			}
		}
	}

	omp_destroy_lock(&qlock);
	return pred;
}

// SSSP sequential baseline
vector<double> dijkstra_seq(AdjList graph, int source) {
	vector<double> dist(graph.size(), INF);
	vector<bool> vis(graph.size(), 0), inQ(graph.size(), 0);
	priority_queue<pair<double, int>> q;

	dist[source] = 0, vis[source] = inQ[source] = 1;
	q.push({ 0, source });

	while (!q.empty()) {
		int u = q.top().second;
		vis[u] = 1, inQ[u] = 0;
		q.pop();

		for (int i = 0; i < graph[u].size(); i++) {
			int v = graph[u][i].first;
			double w = graph[u][i].second + dist[u];

			if (!vis[v] && w < dist[v]) {
				dist[v] = w;
				if (!inQ[v]) {
					inQ[v] = 1;
					q.push({ -w, v });
				}
			}
		}
	}

	return dist;
}

// SSSP parallel implementation 1
vector<double> dijkstra_par(AdjList graph, int source) {
	vector<double> dist(graph.size(), INF);
	vector<bool> vis(graph.size(), 0), inQ(graph.size(), 0);
	priority_queue<pair<double, int>> q;

	omp_lock_t qlock;
	omp_init_lock(&qlock);

	dist[source] = 0, vis[source] = inQ[source] = 1;
	q.push({ 0, source });

	while (!q.empty()) {
		int u = q.top().second;
		vis[u] = 1, inQ[u] = 0;
		q.pop();

		#pragma omp parallel for default(none) shared(dist, vis, inQ, q, qlock, u)
		for (int i = 0; i < graph[u].size(); i++) {
			int v = graph[u][i].first;
			double w = graph[u][i].second + dist[u];

			if (!vis[v] && w < dist[v]) {
				dist[v] = w;
				if (!inQ[v]) {
					inQ[v] = 1;
					omp_set_lock(&qlock);
					q.push({ -w, v });
					omp_unset_lock(&qlock);
				}
			}
		}
	}

	omp_destroy_lock(&qlock);
	return dist;
}

int main(int argc, char *argv[]) {
	int nthreads, mode = 0;
	string input_file;
	if (argc < 3) {
		printf("Usage: %s -n <nthreads> -f <filename>\n", argv[0]);
		exit(1);
	}

	char option;
	while ((option = getopt(argc, argv, "n:f:")) != -1) {
		switch (option) {
		case 'n':
			nthreads = atoi(optarg);
			break;
		case 'f':
			input_file = string(optarg);
			break;
		default:
			printf("Usage: %s -n <nthreads> -f <filename>\n", argv[0]);
			exit(1);
		}
	}

	omp_set_num_threads(nthreads);
	double time = 0.0;

	FILE *input = fopen(input_file.c_str(), "r");
	if (!input) {
		exit(1);
	}

	time = omp_get_wtime();
	int N, M;
	fscanf(input, "%d %d", &N, &M);

	AdjList graph(N);

	for (int i = 0; i < M; i++) {
		int u, v;
		double w;
		fscanf(input, "%d %d %lf", &u, &v, &w);
		graph[u].push_back({ v, w });
	}
	time = omp_get_wtime() - time;
	printf("Setup time (ms): %.6lf\n", time * 1000);

	// TODO: write a helper to run experiments

	// Minimum spanning tree (Prim)
	vector<int> mst_seq, mst_par;

	time = omp_get_wtime();
	for (int i = 0; i < NUM_ITERATIONS; i++) {
		#ifdef _DEBUG
		mst_seq = prim_seq(graph);
		#else
		prim_seq(graph);
		#endif
	}
	time = omp_get_wtime() - time;
	printf("Average sequential MST time (ms): %.6lf\n", time * 1000 / NUM_ITERATIONS);

	time = omp_get_wtime();
	for (int i = 0; i < NUM_ITERATIONS; i++) {
		#ifdef _DEBUG
		mst_par = prim_par(graph);
		#else
		prim_par(graph);
		#endif
	}
	time = omp_get_wtime() - time;
	printf("Average parallel MST time (ms): %.6lf\n", time * 1000 / NUM_ITERATIONS);

	#ifdef _DEBUG
	if (mst_seq != mst_par) {
		fprintf(stderr, "Error: mst results differ\n");
		exit(2);
	}
	#endif

	// Single-source shortest path (Dijkstra)
	int source = 0;
	vector<double> sssp_seq, sssp_par;

	time = omp_get_wtime();
	for (int i = 0; i < NUM_ITERATIONS; i++) {
		#ifdef _DEBUG
		sssp_seq = dijkstra_seq(graph, source);
		#else
		dijkstra_seq(graph, source);
		#endif
	}
	time = omp_get_wtime() - time;
	printf("Average sequential SSSP time (ms): %.6lf\n", time * 1000 / NUM_ITERATIONS);

	time = omp_get_wtime();
	for (int i = 0; i < NUM_ITERATIONS; i++) {
		#ifdef _DEBUG
		sssp_par = dijkstra_par(graph, source);
		#else
		dijkstra_par(graph, source);
		#endif
	}
	time = omp_get_wtime() - time;
	printf("Average parallel SSSP time (ms): %.6lf\n", time * 1000 / NUM_ITERATIONS);

	#ifdef _DEBUG
	if (sssp_seq != sssp_par) {
		fprintf(stderr, "Error: sssp results differ\n");
		exit(2);
	}
	#endif

	fclose(input);
	return 0;
}

#include <algorithm>
#include <cstdio>
#include <omp.h> // Visual Studio's OpenMP is really outdated
#include <queue>
#include <string>
#include <vector>
#include "getopt.h" // For Windows (since no unistd.h)

#define INF 0x3f3f3f3f
#define NUM_ITERATIONS 10 // Number of times to repeat an experiment

using namespace std;

typedef vector<vector<pair<int, double>>> AdjList;
typedef vector<vector<double>> AdjMat;

// MST sequential baseline 1: O(V^2)
vector<double> prim_ser(AdjMat graph) {
	vector<double> dist(graph.size());
	vector<bool> vis(graph.size(), 0);

	dist[0] = 0, vis[0] = 1;
	for (int i = 1; i < graph.size(); i++) {
		dist[i] = graph[0][i];
	}

	for (int n = 1; n < graph.size(); n++) {
		int u = -1;
		double w = INF;

		for (int i = 1; i < graph.size(); i++) {
			if (!vis[i] && dist[i] < w) {
				w = dist[i], u = i;
			}
		}

		vis[u] = 1;
		for (int v = 1; v < graph.size(); v++) {
			if (!vis[v]) {
				dist[v] = min(dist[v], graph[u][v]);
			}
		}
	}

	return dist;
}

// MST parallel implementation 1 (TODO: test this)
vector<double> prim_par(AdjMat graph) {
	vector<double> dist(graph.size());
	vector<bool> vis(graph.size(), 0);

	dist[0] = 0, vis[0] = 1;
	#pragma omp parallel for default(none) shared(dist, graph) schedule(static, 8)
	for (int i = 0; i < graph.size(); i++) {
		dist[i] = graph[0][i]; // schedule(8) avoids false sharing
	}

	//int work = graph.size() / omp_get_max_threads();
	for (int n = 1; n < graph.size(); n++) {
		int u = -1;
		double w = INF;

		// Visual Studio doesn't support user-defined reductions
		#pragma omp parallel default(none) shared(u, w, dist, graph, vis)
		{
			int loc_u = u;
			double loc_w = w;

			#pragma omp for schedule(dynamic) nowait
			for (int i = 1; i < graph.size(); i++) {
				if (!vis[i] && dist[i] < loc_w) {
					loc_w = dist[i], loc_u = i;
				}
			}
			#pragma omp critical // High contention here
			if (loc_w < w) {
				w = loc_w, u = loc_u;
			}
		}

		vis[u] = 1;
		#pragma omp parallel for default(none) shared(dist, graph) schedule(static, 8)
		for (int v = 0; v < graph.size(); v++) {
			if (!vis[v]) {
				dist[v] = min(dist[v], graph[u][v]);
			}
		}
	}

	return dist;
}

// MST sequential baseline 2: O((E + V) log V)
vector<int> prim_ser_q(AdjList graph) {
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

// MST parallel implementation 2
vector<int> prim_par_q(AdjList graph) {
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

		#pragma omp parallel for default(none) shared(dist, vis, pred, q, qlock, u) //schedule(static, 16)
		for (int i = 0; i < graph[u].size(); i++) {
			int v = graph[u][i].first;
			double w = graph[u][i].second;

			if (!vis[v] && w < dist[v]) {
				dist[v] = w, pred[v] = u; // Can't avoid false sharing here
				omp_set_lock(&qlock); // High lock contention
				q.push({ -w, v });
				omp_unset_lock(&qlock);
			}
		}
	}

	omp_destroy_lock(&qlock);
	return pred;
}

// SSSP sequential baseline 2: O((E + V) log V)
vector<double> dijkstra_ser_q(AdjList graph, int source) {
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

// SSSP parallel implementation 2.1
vector<double> dijkstra_par_q(AdjList graph, int source) {
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

		#pragma omp parallel for default(none) shared(dist, vis, inQ, q, qlock, u) //schedule(static, 16)
		for (int i = 0; i < graph[u].size(); i++) {
			int v = graph[u][i].first;
			double w = graph[u][i].second + dist[u];

			if (!vis[v] && w < dist[v]) {
				dist[v] = w; // Can't avoid false sharing here
				if (!inQ[v]) {
					inQ[v] = 1;
					omp_set_lock(&qlock); // High lock contention
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

	FILE *input = fopen(input_file.c_str(), "r");
	if (!input) {
		perror("fopen");
		exit(2);
	}

	int N, M;
	fscanf(input, "%d %d", &N, &M);
	AdjList graph(N);
	AdjMat graph_m(N);

	for (int i = 0; i < N; i++) {
		graph_m[i].resize(N);
	}

	double time = omp_get_wtime();
	// Set up the adjacency list
	for (int i = 0; i < M; i++) {
		int u, v;
		double w;
		fscanf(input, "%d %d %lf", &u, &v, &w);
		graph[u].push_back({ v, w }); // Directed graphs
	}
	time = omp_get_wtime() - time;
	printf("AdjList setup time (ms): %.6lf\n", time * 1000);

	fseek(input, 0, SEEK_SET);
	fscanf(input, "%*d %*d");

	time = omp_get_wtime();
	for (int i = 0; i < M; i++) {
		int u, v;
		double w;
		fscanf(input, "%d %d %lf", &u, &v, &w);
		graph_m[u][v] = w;
	}
	time = omp_get_wtime() - time;
	printf("AdjMat setup time (ms): %.6lf\n", time * 1000);

	// TODO: write a helper to run experiments

	// Minimum spanning tree (Prim)
	vector<double> mst_ser, mst_par;
	vector<int> mst_ser_q, mst_par_q;

	time = omp_get_wtime();
	for (int i = 0; i < NUM_ITERATIONS; i++) {
		#ifdef _DEBUG
		mst_ser = prim_ser(graph_m);
		#else
		prim_ser(graph_m);
		#endif
	}
	time = omp_get_wtime() - time;
	printf("Avg O(V^2) ser MST time (ms): %.6lf\n", time * 1000 / NUM_ITERATIONS);

	time = omp_get_wtime();
	for (int i = 0; i < NUM_ITERATIONS; i++) {
		#ifdef _DEBUG
		mst_par = prim_par(graph_m);
		#else
		prim_par(graph_m);
		#endif
	}
	time = omp_get_wtime() - time;
	printf("Avg O(V^2) par MST time (ms): %.6lf\n", time * 1000 / NUM_ITERATIONS);

	#ifdef _DEBUG
	if (mst_ser != mst_par) {
		fprintf(stderr, "Error: mst results differ\n");
		exit(3);
	}
	#endif

	time = omp_get_wtime();
	for (int i = 0; i < NUM_ITERATIONS; i++) {
		#ifdef _DEBUG
		mst_ser_q = prim_ser_q(graph);
		#else
		prim_ser_q(graph);
		#endif
	}
	time = omp_get_wtime() - time;
	printf("Avg O((E+V)log(V)) ser MST time (ms): %.6lf\n", time * 1000 / NUM_ITERATIONS);

	time = omp_get_wtime();
	for (int i = 0; i < NUM_ITERATIONS; i++) {
		#ifdef _DEBUG
		mst_par_q = prim_par_q(graph);
		#else
		prim_par_q(graph);
		#endif
	}
	time = omp_get_wtime() - time;
	printf("Avg O((E+V)log(V)) par MST time (ms): %.6lf\n", time * 1000 / NUM_ITERATIONS);

	#ifdef _DEBUG
	if (mst_ser_q != mst_par_q) {
		fprintf(stderr, "Error: mst results differ\n");
		exit(3);
	}
	#endif

	// Single-source shortest path (Dijkstra)
	int source = 0;
	vector<double> sssp_ser, sssp_par;

	time = omp_get_wtime();
	for (int i = 0; i < NUM_ITERATIONS; i++) {
		#ifdef _DEBUG
		sssp_ser = dijkstra_ser_q(graph, source);
		#else
		dijkstra_ser_q(graph, source);
		#endif
	}
	time = omp_get_wtime() - time;
	printf("Avg O((E+V)log(V)) ser SSSP time (ms): %.6lf\n", time * 1000 / NUM_ITERATIONS);

	time = omp_get_wtime();
	for (int i = 0; i < NUM_ITERATIONS; i++) {
		#ifdef _DEBUG
		sssp_par = dijkstra_par_q(graph, source);
		#else
		dijkstra_par_q(graph, source);
		#endif
	}
	time = omp_get_wtime() - time;
	printf("Avg O((E+V)log(V)) par SSSP time (ms): %.6lf\n", time * 1000 / NUM_ITERATIONS);

	#ifdef _DEBUG
	if (sssp_ser != sssp_par) {
		fprintf(stderr, "Error: sssp results differ\n");
		exit(3);
	}
	#endif

	fclose(input);
	return 0;
}

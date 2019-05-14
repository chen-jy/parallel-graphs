#include <cstdio>
#include <errno.h>
#include <queue>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>
#include "getopt.h"

#define W_LB 0.0
#define W_UB 1.0

using namespace std;

struct pipid_hash {
	inline size_t operator()(const pair<int, pair<int, double>> &v) const {
		return v.first * 10001 + v.second.first;
	}
};

int randint(int lb, int ub) {
	random_device dev;
	mt19937 rng(dev());
	uniform_int_distribution<mt19937::result_type> dist(lb, ub);
	return dist(rng);
}

double randdouble(double lb, double ub) {
	random_device dev;
	mt19937 rng(dev());
	uniform_real_distribution<double> dist(lb, ub);
	return dist(rng);
}

void dfs(vector<unordered_set<int>> graph, vector<int> *colour, int x, int c) {
	(*colour)[x] = c;
	for (auto it = graph[x].begin(); it != graph[x].end(); ++it) {
		if ((*colour)[*it] == -1) {
			dfs(graph, colour, *it, c);
		}
	}
}

void bfs(vector<unordered_set<int>> graph, vector<int> *colour, int x, int c) {
	queue<int> q;
	q.push(x);

	while (!q.empty()) {
		x = q.front(), q.pop();
		(*colour)[x] = c;

		for (auto it = graph[x].begin(); it != graph[x].end(); ++it) {
			if ((*colour)[*it] == -1) {
				q.push(*it);
			}
		}
	}
}

int main(int argc, char *argv[]) {
	int N, M, d, c;
	string output_file;
	if (argc < 6) {
		printf("Usage: %s -v <vertices> -e <edges> -d <directed> -c <connected> -o <filename>\n", argv[0]);
		exit(1);
	}

	char option;
	while ((option = getopt(argc, argv, "v:e:d:c:o:")) != -1) {
		switch (option) {
		case 'v':
			N = atoi(optarg);
			break;
		case 'e':
			M = atoi(optarg);
			break;
		case 'd':
			d = atoi(optarg);
			if (d != 0 && d != 1) {
				printf("Error: d must be either 0 or 1\n");
				exit(1);
			}
			break;
		case 'c':
			c = atoi(optarg);
			if (c != 0 && c != 1) {
				printf("Error: c must be either 0 or 1\n");
				exit(1);
			}
			if (c == 1 && d == 1) {
				// Since we cannot guarantee a forest of DAGs are produced if d = 1
				printf("Error: the graph may only be connected if undirected\n");
				exit(1);
			}
			break;
		case 'o':
			output_file = string(optarg);
			break;
		default:
			printf("Usage: %s -v <vertices> -e <edges> -d <directed> -c <connected> -o <filename>\n", argv[0]);
			exit(1);
		}
	}

	// Generate the graph
	printf("Generating graph...\n");
	vector<unordered_set<int>> graph(N);
	unordered_set<pair<int, pair<int, double>>, pipid_hash> edges; // Random ordering

	for (int i = 0; i < M; i++) {
		int u = randint(0, N - 1);
		int v = randint(0, N - 1);
		double w = randdouble(W_LB, W_UB);
		
		if (u == v || graph[u].find(v) != graph[u].end()) {
			i--;
			continue;
		}

		graph[u].insert(v);
		if (!d) graph[v].insert(u);
		// Don't need to insert both edges (if undirected) since that's up to the user's algorithm
		edges.insert({ u, {v, w} });
	}

	// If the undirected graph needs to be connected
	if (!d && c) {
		printf("Connecting graph...\n");
		vector<int> colour(N, -1);
		int col = 0;

		for (int i = 0; i < N; i++) {
			if (colour[i] == -1) {
				//dfs(graph, &colour, i, col++);
				bfs(graph, &colour, i, col++);
			}
		}

		// Connect all forests to vertex 0, for simplicity
		unordered_set<int> connected;
		for (int i = 1; i < N; i++) {
			if (colour[i] != colour[0] && connected.find(colour[i]) == connected.end()) {
				edges.insert({ 0, {i, randdouble(W_LB, W_UB)} });
				connected.insert(colour[i]);
			}
		}

		if (col != 1) {
			M += col - 1;
			printf("Note: added %d edges to connect undirected graph\n", col - 1);
		}
	}

	// Print the graph to the file
	printf("Printing graph...\n");
	FILE *fd = fopen(output_file.c_str(), "w");
	if (!fd) {
		perror("fopen");
		exit(1);
	}

	fprintf(fd, "%d %d\n", N, M);
	for (auto it = edges.begin(); it != edges.end(); ++it) {
		fprintf(fd, "%d %d %lf\n", it->first, it->second.first, it->second.second);
	}

	fclose(fd);
	return 0;
}

#include <cstdio>
#include <errno.h>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>
#include "getopt.h"

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

int main(int argc, char *argv[]) {
	int N, M;
	string output_file;
	if (argc < 4) {
		printf("Usage: %s -v <vertices> -e <edges> -o <filename>\n", argv[0]);
		exit(1);
	}

	char option;
	while ((option = getopt(argc, argv, "v:e:o:")) != -1) {
		switch (option) {
		case 'v':
			N = atoi(optarg);
			break;
		case 'e':
			M = atoi(optarg);
			break;
		case 'o':
			output_file = string(optarg);
			break;
		default:
			printf("Usage: %s -v <vertices> -e <edges> -o <filename>\n", argv[0]);
			exit(1);
		}
	}

	// Generate the graph
	vector<unordered_set<int>> graph(N);
	unordered_set<pair<int, pair<int, double>>, pipid_hash> edges; // Random ordering

	for (int i = 0; i < M; i++) {
		int u = randint(0, N - 1);
		int v = randint(0, N - 1);
		double w = randdouble(0.0, 1.0);
		
		if (u == v || graph[u].find(v) != graph[u].end()) {
			i--;
			continue;
		}

		graph[u].insert(v);
		edges.insert({ u, {v, w} });
	}

	// Print the graph to the file
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

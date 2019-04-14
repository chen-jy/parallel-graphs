#include <cstdio>
#include <list>
#include <omp.h>
#include <string>
#include <vector>
#include "getopt.h"

// Process input in an online/offline style
#define ONLINE 0
#define OFFLINE 1

using namespace std;

int main(int argc, char *argv[]) {
	int tid, nthreads, mode;
	string input_file;
	if (argc < 4) {
		printf("Usage: %s -n <nthreads> -f <filename> -m <mode>\n", argv[0]);
		exit(1);
	}

	char option;
	while ((option = getopt(argc, argv, "n:f:m:")) != -1) {
		switch (option) {
		case 'n':
			nthreads = atoi(optarg);
			break;
		case 'f':
			input_file = string(optarg);
			break;
		case 'm':
			mode = atoi(optarg);
			break;
		default:
			printf("Usage: %s -n <nthreads> -f <filename>\n", argv[0]);
			exit(1);
		}
	}

	omp_set_num_threads(nthreads);

	FILE *input = fopen(input_file.c_str(), "r");
	int N, M;
	fscanf(input, "%d %d", &N, &M);

	vector<list<pair<int, double>>> graph(N);

	if (mode == ONLINE) {
		for (int i = 0; i < M; i++) {
			int u, v;
			double w;
			fscanf(input, "%d %d %lf", &u, &v, &w);
			graph[u].push_back({ v, w });
		}
	}
	else if (mode == OFFLINE) {
		vector<list<pair<int, double>>> gp(N);

		#pragma omp parallel default(none) private(tid, gp) shared(nthreads, N, M, input, graph)
		{
			tid = omp_get_thread_num();
			int nlines = M / nthreads, lnlines = nlines;
			if (tid == nthreads - 1) {
				lnlines = M - nlines * tid;
			}

			FILE *fd = input + tid * nlines;
			#pragma omp for
			for (int i = 0; i < lnlines; i++) {
				int u, v;
				double w;
				fscanf(fd, "%d %d %lf", &u, &v, &w);
				gp[u].push_back({ v, w });
			}

			// Visual Studio doesn't support custom reductions
			#pragma omp critical
			{
				for (int i = 0; i < N; i++) {
					graph[i].splice(graph[i].end(), gp[i]);
				}
			}
		}
	}
	else exit(2);

	return 0;
}

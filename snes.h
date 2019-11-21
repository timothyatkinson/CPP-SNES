#ifndef SNES_H
#include <vector>
using namespace std;

vector<double> separable_natural_evolution_strategies(vector<double> mu_0, vector<double> sigma_0, double (*fitness_function)(vector<double>), int generations, int update_rate);
void print_vector(vector<double> vec);
#endif

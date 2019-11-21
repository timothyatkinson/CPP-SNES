#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include "snes.h"
using namespace std;

static int dims = 10;
static vector<double> target;
double ellipse_fitness(vector<double> candidate){
  double sum = 0.0;
  for(int i = 0; i < dims; i++){
    sum += fabs(candidate[i] - target[i]);
  }
  return sum;
}

int main(){


  srand(time(NULL));

  //Our initial mu is 0s, sigma is 1s, and our target vector is [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
  vector<double> mu_0;
  vector<double> sigma_0;
  for(int i = 0; i < dims; i++){
    mu_0.push_back(0.0);
    sigma_0.push_back(1.0);
    target.push_back((i + 1) * 2);
  }

  //We're using ellipse_fitness, e.g. absolute distance between mu and the target vector
  //Run for 100 generations and print updates every 25 generations
  vector<double> mu_final = separable_natural_evolution_strategies(mu_0, sigma_0, ellipse_fitness, 100, 25);

  //Some outputs
  cout << "Initial mu:\n";
  print_vector(mu_0);
  cout << "Initial sigma:\n";
  print_vector(sigma_0);
  cout << "Target mu:\n";
  print_vector(target);
  cout << "Final mu:\n";
  print_vector(mu_final);

  //Clean up
  mu_0.clear();
  sigma_0.clear();
  mu_final.clear();
}

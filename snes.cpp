#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <assert.h>
#include <bits/stdc++.h>
#include "snes.h"
using namespace std;

void print_vector(vector<double> vec){
  int length = vec.size();
  cout << "[";
  for(int i = 0; i < length; i++){
    if(i > 0){
      cout << ", ";
    }
    cout << vec[i];
  }
  cout << "]\n";
}

//This utility function is taken from "Natural Evolution Strategies", Wierstra et. al., 2011.
vector<double> utility_function(vector<double> fitness_values, int lambda){
  //First we'll sort the fitness values
  vector<double> sorted_fitness = fitness_values;
  sort(sorted_fitness.begin(), sorted_fitness.end());
  vector<double>::iterator it;
  vector<double> utility;

  //Next we'll find the rank of each individual and compute its utilty.
  for(int l = 0; l < lambda; l++){
    double f_l = fitness_values[l];
    it = find(sorted_fitness.begin(), sorted_fitness.end(), f_l);
    int rank = it - sorted_fitness.begin() + 1;
    double utlity_sum = 0.0;
    for(int i = 1; i <= lambda; i++){
      utlity_sum += max(0.0, log2(((double)lambda/2) + 1.0) - log2((double)i));
    }
    utility.push_back(max(0.0, log2(((double)lambda/2) + 1.0) - log2((double)rank)) / utlity_sum - (1.0 / (double)lambda));
  }
  sorted_fitness.clear();

  //Let's do some cleanup
  return utility;
}

/*For a general overview of the algorithm, see "High Dimensions and Heavy Tails for Natural Evolution Strategies", Schaul et. al. 2011.
The algorithm is as described in that paper. For the utility function, see utility_function defined above.
We also use the 'generic' parameters given in the paper.
The arguments are:
  vector<double> mu_0 - the initial value of mu.
  vector<double> sigma_0 - the initial value of sigma.
  double (*fitness_function)(vector<double>) - the fitness function. Takes a vector as input and returns a double as output.
  int generations - the number of generations to run SNES for.
  int update_rate - how often the function should tell you its progress. Set to <= 0 for no updates.
Returns:
  vector<double> mu - the vector at the end of the evolutionary run.
*/
vector<double> separable_natural_evolution_strategies(vector<double> mu_0, vector<double> sigma_0, double (*fitness_function)(vector<double>), int generations, int update_rate){
  int vector_length = mu_0.size();
  assert(vector_length == sigma_0.size());

  //We won't modify or free mu_0 or sigma_0.
  vector<double> mu = mu_0;
  vector<double> sigma = sigma_0;

  //Load a normal distribution with mean 0 and std. dev 1
  default_random_engine generator;
  normal_distribution<double> gaussian(0.0, 1.0);

  //Default parameters.
  int lambda = 4 + (int)floor(3 * log2(vector_length));
  double learning_rate_mu = 1.0;
  double learning_rate_sigma = (3 + log2(vector_length)) / (5 * sqrt(vector_length));

  cout << "Running SNES with lambda = " << lambda << " learning_rate_mu = " << learning_rate_mu << " learning_rate_sigma = " << learning_rate_sigma << "\n";

  //Repeat until all generations are finished.
  /* In each generation, SNES performs the following steps:
    1. Generate lambda normal distribution vectors, s.
    2. Compute the children, z, by adding members of s (scaled by sigma) to mu.
    3. Evaluate the children.
    4. Compute utilities for the children based on their fitness.
    5. Compute the natural gradients (under the assumption of separable fitness).
    5. Update mu and sigma based on the natural gradients.
  */
  for(int g = 0; g < generations; g++){

    //We're going to generate lambda children and assign to each a fitness value.
    vector< vector<double> > s;
    vector<double> fitness_values;

    //1. GENERATE, 2. BUILD CHILDREN and 3. EVALUTE CHILDREN can all be done in this one loop.
    //Each child is literally a vector of samples from gaussian
    for(int k = 0; k < lambda; k++){
      //(Briefly) construct and evaluate z_k;
      vector<double> s_k;
      vector<double> z_k;
      s.push_back(s_k);
      for(int x = 0; x < vector_length; x++){
        //Generate a random sample from gaussian.
        double s_k_x = gaussian(generator);
        s[k].push_back(s_k_x);

        //Fill in z_k as we go
        double z_k_x = mu[x] + (s_k_x * sigma[x]);
        z_k.push_back(z_k_x);
      }
      //Evaluate z_k
      fitness_values.push_back(fitness_function(z_k));
      z_k.clear();
    }

    //4. COMPUTE UTILITIES
    //Now we're going to transform our fitness values into utility values.
    vector<double> utility_values = utility_function(fitness_values, lambda);

    //5. COMPUTE NATURAL GRADIENTS and 6. UPDATE MU AND SIGMA
    //These are on the assumption that the gradients are seperable so we needn't bother with covariance.
    //For simplicity we're going to fold the learning rates into the gradients

    for(int x = 0; x < vector_length; x++){
      //Update the xth element of mu_gradient
      double mu_gradient = 0.0;
      for(int l = 0; l < lambda; l++){
        mu_gradient += utility_values[l] * s[l][x];
      }
      mu[x] += (learning_rate_mu * sigma[x] * mu_gradient);
      //Update the xth element of the sigma_gradient
      double sigma_gradient = 0.0;
      for(int l = 0; l < lambda; l++){
        sigma_gradient += utility_values[l] * (s[l][x] * s[l][x] - 1.0);
      }
      sigma[x] = (sigma[x] * exp((learning_rate_sigma / 2.0) * sigma_gradient));
    }

    //6. UPDATE MU AND SIGMA

    if(update_rate > 0 && g % update_rate == 0){
      cout << "Generation " << g << "\n";
      cout << "Mu fitness " << fitness_function(mu) << "\n";
      cout << "Fitness values\n";
      print_vector(fitness_values);
      cout << "Utility values\n";
      print_vector(utility_values);
      cout << "Mu\n";
      print_vector(mu);
      cout << "Sigma\n";
      print_vector(sigma);
    }

    //Make sure we're clearing out all of the memory associated with the children.
    for(int k = 0; k < lambda; k++){
      s[k].clear();
    }
    //Free everything that's not permanent.
    s.clear();
    fitness_values.clear();
    utility_values.clear();
  }
  //We don't need sigma any more.
  sigma.clear();
  return mu;
}

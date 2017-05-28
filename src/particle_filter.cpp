/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <map>

#include "particle_filter.h"


#define DEBUG 0
using namespace std;


/**
 * print_particle: Print the particle as a X,Y,theta tuple
 */
void print_particle(Particle p)
{
	std::printf(" Particle (X,Y,theta) is (%.2f,%.2f,%.2f) \n", p.x, p.y, p.theta);
}

/**
 * Init: Initialize Particles
 * @params: (x,y,theta) position and array of standard deviations
 */
void ParticleFilter::init(double x, double y, double theta, double std[])
{
	//Initialize the Random number Generator
	default_random_engine gen;

	// Initialize Probability distributions based on the standard deviations
	normal_distribution<double> x_dist(x, std[0]);
	normal_distribution<double> y_dist(y, std[1]);
	normal_distribution<double> theta_dist(theta, std[2]);

	//Seed Particles  to a reasonable number
	// Personal TODO: Adapt this code segment to using AMCL using the KDL criteria
	num_particles = 50;


	// Resize the particles weights vectors
	particles.resize(num_particles);
	weights.resize(num_particles);

	// Seed particles based on distribution
	// Using this approach because it can be parallelized
	// at a later point using omp or tbb while push_back based approach can't
	for (int i = 0; i < particles.size(); i++)
	{
		// Create Particle
		Particle p_temp;
		// Create Id
		p_temp.id = i;

		// Draw from distributions created earlier
		p_temp.x = x_dist(gen);
		p_temp.y = y_dist(gen);
		p_temp.theta = theta_dist(gen);

		// Set weights to 1.0
		// Slightly confused here. Shouldn't all weights be set to 1/num_particles
		// since all particles are equally likely during the initialization and their weights should sum
		// up to one to maintain them as a valid probability distribution
		p_temp.weight = 1.0;

		if (DEBUG)
			print_particle(p_temp);

		// copy p_temp into the vector at index i
		particles[i] = p_temp;
	}

	// Setting flag to indicate initialization completed
	is_initialized = true;

}

/**
 * Prediction: Predict the Position of all particles based on timestep and control input
 * @param delta_t: timestep
 * @param std_pos: standard deviations of x,y,theta
 * @param velocity: v component of control input
 * @param yaw_rate: w component of control input
 */
void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
	// Initialize Random number generator This should probably be a class variable since its being used so much
	// But I've heard on slack that changing the header file at all leads to a cannot grade.
	default_random_engine gen;

	// create x,y,theta variables
	double x, y, theta;

	for (int i = 0; i < particles.size(); i++)
	{
		Particle& p_temp = particles[i];

		// Apply prediction equation based on whether delta_yaw rate to zero
		if (fabs(yaw_rate) < 0.0001)
		{
			double vdt = velocity*delta_t;
			x = p_temp.x + vdt*cos(p_temp.theta);
			y =  p_temp.y + vdt*sin(p_temp.theta);
			theta = p_temp.theta;
		}
		else
		{
			// Computing once so that we don't have to waste flops
			double k = velocity/yaw_rate;
			double ydt = yaw_rate*delta_t;

			x = p_temp.x + k*(sin(p_temp.theta + ydt) - sin(p_temp.theta));
			y = p_temp.y + k*(cos(p_temp.theta) - cos(p_temp.theta + ydt));
			theta = ydt + p_temp.theta;
		}

		// Add Noise to the poisitions
		normal_distribution<double> x_dist(x, std_pos[0]);
		normal_distribution<double> y_dist(y, std_pos[1]);
		normal_distribution<double> theta_dist(theta, std_pos[2]);

		// Assign position to temporary particle
		p_temp.x = x_dist(gen);
		p_temp.y = y_dist(gen);
		p_temp.theta = theta_dist(gen);

		if (DEBUG)
			print_particle(p_temp);
	}

}

/**
 * dataAssociation: Associate closest landmark to observations
 * @param predicted: set of predicted/Expected Landmarks
 * @param observations: set of actual observations
 */
void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations)
{
	// Go through observations
	for (int i = 0; i < observations.size(); i++)
	{
		// Set a default value to use for comparison
		double  min_dist = 100000000.0;
		LandmarkObs& observation = observations[i];

		for ( int j = 0; j < predicted.size(); j++)
		{
			LandmarkObs& pred = predicted[j];

			// Calculate the distance between ith observation and jth prediction
			double distance = dist(pred.x, pred.y, observation.x, observation.y);

			// Change the Id if distance is smaller than the current minimum value
			// This is mn complex
			// Personal TODO: Read up on more approaches to the association problem
			if (distance < min_dist)
			{
				min_dist = distance;
				observation.id = pred.id;
			}
		}
	}

}

/**
 * updateWeights: Update step of the PF which updates the weights based on observations and actual landmark positions
 * @param sensor_range: range of the sensor
 * @param std_landmark: standard deviations in landmark x and y
 * @param observations: observations
 * @param Map of landmarks
 */
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks)
{
	// Create a TOBS vector after transforming the position
	std::vector<LandmarkObs> t_obs;
	t_obs.resize(observations.size());


	// Iterate through particles
	for (int i=0; i < particles.size(); i++)
	{
		Particle &particle = particles[i];


		// Iterate through observations
		for (int j=0; j < observations.size(); j++)
		{
			LandmarkObs t_ob;

			// transform from vehicle coordinate to map coordinate
			t_ob.x = observations[j].x * cos(particle.theta) - observations[j].y * sin(particle.theta) + particle.x;
			t_ob.y = observations[j].x * sin(particle.theta) + observations[j].y * cos(particle.theta) + particle.y;

			t_obs[j] = t_ob;
		}

		// Vector to hold adjacent landmarks
		std::vector<LandmarkObs> observable_landmarks;

		// Create a mapping between landmarks and map objects for easy lookup
		std::map<int, Map::single_landmark_s> id_landmark_map;

		// For every landmark check if observation is in range
		for(int k = 0; k < map_landmarks.landmark_list.size(); k++)
		{
			Map::single_landmark_s& landmark = map_landmarks.landmark_list[k];

			// For each landmark check distance
			double distance = dist(landmark.x_f, landmark.y_f, particle.x, particle.y);

			// If Landmark is in range of sensor
			if (distance <= sensor_range)
			{
				// If it is in range add to the
				observable_landmarks.push_back(LandmarkObs{ landmark.id_i,landmark.x_f,landmark.y_f });

				// Add the landmark to the mapping we defined earlier for easy lookup
				id_landmark_map.insert(std::make_pair(landmark.id_i, landmark));
			}
		}

		// For all observable landmarks
		if (observable_landmarks.size() > 0)
		{
			// get the association of the landmark to the transformed obstacle
			dataAssociation(observable_landmarks, t_obs);

			// ReInitialize weight to 1
			particle.weight = 1.0;

			for (int l = 0; l < t_obs.size(); l++)
			{
				LandmarkObs observation = t_obs[l];
				// update the particle weight

				// Expected position
				double xu = id_landmark_map[observation.id].x_f;
				double yu = id_landmark_map[observation.id].y_f;

				// Observed position
				double x = observation.x;
				double y = observation.y;

				// standard_deviations
				double std_x = std_landmark[0];
				double std_y = std_landmark[1];

				// Calculate the Multivariate gaussian probability
				double x_diff = (x - xu) * (x -xu) / (2 * std_x * std_x);
				double y_diff = (y - yu) * (y - yu) / (2 * std_y * std_y);
				particle.weight *= 1.0 / (2 * M_PI * std_x * std_y) * exp(-(x_diff + y_diff));

				if(DEBUG)
				std::cout << "Weight of particle"<<i <<"is" << particle.weight << std::endl;
			}
			weights[i] = particle.weight;

		}
		else
		{
			weights[i] = 0.0;
		}
	}
}

/**
 * resample: resample based on discrete distribution
 * This function concentrates the particles in the highest posterior likelihood region
 */
void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// Use Random device
	std::random_device random;

	// Create a Discrete distribution
	std::mt19937 gen(random());
	std::discrete_distribution<> dist(weights.begin(), weights.end());

	// Create new particles
	std::vector<Particle> particles_resampled;
	particles_resampled.resize(num_particles);

	// resample based on weights
	for (int i = 0; i < num_particles; i++)
	{
		particles_resampled[i] = particles[dist(gen)];

		if (DEBUG)
			print_particle(particles_resampled[i]);
	}

	// Reassign particles to resampled
	particles = particles_resampled;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length()-1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
	copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length()-1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length()-1);  // get rid of the trailing space
	return s;
}

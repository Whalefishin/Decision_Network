#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <utility>
#include <limits>

#include "Neuron.h"
#include "Statistics.h"


using namespace std;

class Network{
    public:
    Network(int num_neurons, double w, double lambda, double time_step);
    Network(int num_neurons, double w, double lambda, double time_step, int norm);
    Network(int num_neurons, double w, double lambda, double time_step, int norm,int k);
    ~Network();

    void insertUndirectedConnection(Neuron* n1, Neuron* n2);
    void insertUndirectedConnection(int n1, int n2);
    void removeUndirectedConnection(Neuron* n1, Neuron* n2);
    void removeUndirectedConnection(int n1, int n2);
    void insertUndirectedConnectionNoChecking(Neuron* n1, Neuron* n2); // use this at your own risk
    void insertUndirectedConnectionNoChecking(int n1, int n2);
    bool isConnected(Neuron* n1, Neuron* n2);
    bool isConnected(int n1, int n2);
    vector<Neuron*> getNeighbors(Neuron* n);

    void update(double (Neuron::*f)(double));
    void update(double (Neuron::*f)(double),int c); //update the whole network one step forward
    void updateIntegrateAll(double (Neuron::*f)(double)); //gain fcn incoporates evidence
    void updateWithChoice(int c, int g);
    void computeAccuracy();
    void computeAccuracy(int k);

    void initializeFairICYesDist(double IC, double diff);
    void initializeFairICNoDist(double IC, double diff);
    void initializeRandomICYesDist(double diff);
    void initializeRandomICNoDist(double diff); //use this for schematic initialization. diff is the difference in S btw winner and others
    void initializeWithChoice(int c1, int c2, double IC, double diff);
    void constructRegularNetwork(int k); //k is the mean degree, assumed to be even
    void constructRandomNetwork(double p); //p is the prob that any given possible edge exists
    void constructSmallWorldNetwork(int k, double p); //p is the rewiring prob.
    void constructAllToAllNetwork();

    double computeCharacteristicPathLength();
    double computeGlobalClusteringCoefficient();

    int** outputAdjacencyMtx();

    //fields
    int num_neurons;
    vector<Neuron*> neuron_vector; //contains all the neurons
    vector<Neuron*> winners;

    double time;
    double time_step;

    double RT; //reaction time
    double Acc; //Accuracy

};

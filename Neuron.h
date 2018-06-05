#include <string>
#include <iostream>
#include <utility>
#include <vector>
#include <math.h>
#include <ctime>

#include "Statistics.h"

class Neuron{
    public:
    Neuron(int number, double S, double lambda, double w, 
    double x_0, double t_0, double h);

    ~Neuron();

    void updateRK4(double (Neuron::*f)(double)); //parameter specifies which activiation fcn to use.
    double computeRHS(double t, double x,double (Neuron::*f)(double));

    void updateRK4IntegrateAll(double (Neuron::*f)(double)); //difference is if gain fcn incoporates S, the evidence
    double computeRHSIntegrateAll(double t, double x, double (Neuron::*f)(double));
    
    double linearActiv(double x); //linear activation function
    double sigmActiv(double x); //sigmodal activation funciton
    double rectLinearActiv(double x); //linear on the positive side, zero otherwise.
    double binaryActiv(double x);

    //fields
    int number; //metadata, for the record
    vector<Neuron*> neighbors; //adjacent neurons
    int update_count; //amount of times this neuron has been updated

    double S; //evidence
    double lambda; //self-inhibition constant
    double w; //cross-inhibition constant
    double x; //decision variable for this neuron, dependent variable.
    double t; //time, the independent variable.
    double x_prev;
    double t_prev;

    double h; //step for RK4

    //for reaction time
    double RT_sum; //sum for checking equilibrium
    double RT_threshold; //threshold that the sum compares to
    vector<double> RT_vector; //for efficiency
    int RT_history; //number of history wanted.
    bool RT_collected;
    int RT_Count;

    vector<double> x_data;
    vector<double> t_data;




};
#include "Neuron.h"

using namespace std;

Neuron::Neuron(int number, double S, double lambda, double w, 
double x_0, double t_0, double h){
    this->number = number;
    x = x_0;
    x_prev = x_0;
    t = t_0;
    t_prev = t_0;
    this->S = S;
    this->lambda = lambda;
    this->w = w;
    this->h = h;

    update_count =0;
    RT_sum = 0;
    RT_threshold = 0.001;
    RT_history = 200;
    RT_collected = false;
    RT_Count = 0;

    x_data.push_back(x);
    t_data.push_back(t);
}

// Neuron::~Neuron(){

// }


void Neuron::updateRK4(double (Neuron::*f)(double)){

    update_count++;

    double k_1 = h * computeRHS(this->t, this->x, (f));
    double k_2 = h * computeRHS(this->t + (h/2.0), this->x + (k_1/2.0), (f));
    double k_3 = h * computeRHS(this->t + (h/2.0), this->x + (k_2/2.0), (f));
    double k_4 = h * computeRHS(this->t + (h), this->x + (k_3), (f));

    //data collection for Reaction Time
    // if (!RT_collected){
    //     RT_sum += fabs((k_1 + 2*k_2 + 2*k_3 + k_4)/6.0);
    //     //RT_vector.push_back(fabs((k_1 + 2*k_2 + 2*k_3 + k_4)/6.0));
    //     if (update_count > RT_history){
    //         //RT_sum -= RT_vector[0];
    //         //RT_vector.erase(RT_vector.begin());
    //     }
    // }

    if (!RT_collected){
        double d = fabs((k_1 + 2*k_2 + 2*k_3 + k_4)/6.0);
        if (d < RT_threshold){
            RT_Count++;
        }
        else{
            RT_Count = 0;
        }
    }

    if (number ==0){
        cout << "K_1: " + to_string(k_1) << endl;
        cout << "K_2: " + to_string(k_2) << endl;
        cout << "K_3: " + to_string(k_3) << endl;
        cout << "K_4: " + to_string(k_4) << endl;
        
        
        //cout << (k_1 + 2*k_2 + 2*k_3 + k_4)/6.0 << endl;
    }

    //updating
    t_prev = t;
    x_prev = x;
    t += h;
    x += (k_1 + 2*k_2 + 2*k_3 + k_4)/6.0; 

    //recording
    x_data.push_back(x);
    t_data.push_back(t);
}


double Neuron::computeRHS(double t, double x,double (Neuron::*f)(double)){
    double ret = 0;
    double integratedSum = 0;

    for (int i=0; i<neighbors.size() ;i++){
        Neuron* n = neighbors[i];
        integratedSum += (this->*f)(n->x_prev);
    }

    ret = rectLinearActiv(S - w * integratedSum) - lambda * x;
    //cout << lambda * x << endl;
    if (number == 0){
        cout << "x: " + to_string(x) <<endl;
        cout << "inhibition term: " + to_string(w * integratedSum) << endl;
        cout << "Integrated: " + to_string(rectLinearActiv(S - w * integratedSum)) << endl;
        cout << "ret: " + to_string(ret) << endl;
    }

    return ret;    
}

void Neuron::updateRK4IntegrateAll(double (Neuron::*f)(double)){
    update_count++;

    double k_1 = h * computeRHSIntegrateAll(this->t, this->x, (f));
    double k_2 = h * computeRHSIntegrateAll(this->t + (h/2.0), this->x + (k_1/2.0), (f));
    double k_3 = h * computeRHSIntegrateAll(this->t + (h/2.0), this->x + (k_2/2.0), (f));
    double k_4 = h * computeRHSIntegrateAll(this->t + (h), this->x + (k_3), (f));

    // if (!RT_collected){
    //     RT_sum += fabs((k_1 + 2*k_2 + 2*k_3 + k_4)/6.0);
    //     RT_vector.push_back(fabs((k_1 + 2*k_2 + 2*k_3 + k_4)/6.0));
    //     if (update_count > RT_history){
    //         RT_sum -= RT_vector[0];
    //         RT_vector.erase(RT_vector.begin());
    //     }
    // }

    if (!RT_collected){
        double d = fabs((k_1 + 2*k_2 + 2*k_3 + k_4)/6.0);
        if (d < RT_threshold){
            RT_Count++;
        }
        else{
            RT_Count = 0;
        }
    }

    //updating
    t_prev = t;
    x_prev = x;
    t += h;
    x += (k_1 + 2*k_2 + 2*k_3 + k_4)/6.0; 

    //recording
    x_data.push_back(x);
    t_data.push_back(t);
}


double Neuron::computeRHSIntegrateAll(double t, double x, double (Neuron::*f)(double)){
    double ret = 0;
    double toIntegrate =0; //to be put inside the gain fcn
    for (int i=0;i<neighbors.size();i++){
        Neuron* n = neighbors[i];
        toIntegrate += n->x_prev;
    }
    ret = - lambda * x + (this->*f)(-w * toIntegrate + S);
    return ret;
}

double Neuron::linearActiv(double x){
    return x;
}

double Neuron::sigmActiv(double x){
    //parameters
    double a = 0.5;
    double k = 4;
    double ret;
    ret = 1.0/(1+exp(-k*(x-a)));
    return ret;
}

double Neuron::rectLinearActiv(double x){
    return max(x,0);
}

double Neuron::binaryActiv(double x){
    //parameter
    double threshold = 0.4;
    if (x > threshold){
        return 1;
    }
    else{
        return 0;
    }
}
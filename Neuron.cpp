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
    RT_threshold = 0.00001;
    mean_threshold = 0.000000001;
    RT_history = 200;
    RT_collected = false;
    RT_Count = 0;
    jump_ratio = 10;
    jump_variation_allowance = 0.1;
    prev_jump_peak = 0;
    x_threshold = 0.8;

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

    // if (!RT_collected){
    //     double d = (k_1 + 2*k_2 + 2*k_3 + k_4)/6.0;
    //     RT_sum += d;
    //     d_history.push_back(d);
    //     if (update_count >=RT_history){
    //         RT_sum -= d_history[0];
    //         d_history.erase(d_history.begin());
    //     }
    //     if (fabs(d) < RT_threshold){
    //         RT_Count++;
    //     }
    //     else{
    //         if (RT_sum/d_history.size() > mean_threshold){ //mean is fluctuation
    //             RT_Count = 0;
    //         }
    //     //   double d_prev = fabs(x_data[x_data.size()-1] - x_data[x_data.size()-2]);
    //     //   if (d/d_prev >jump_ratio){ //this is a jump
    //     //     double jump_peak = x+d;
    //     //     if (fabs(jump_peak-prev_jump_peak) > jump_variation_allowance && prev_jump_peak !=0){ //jumps are changing their peaks
    //     //       RT_Count = 0;
    //     //     }
    //     //     prev_jump_peak = jump_peak;
    //     //   }
    //     //   else{ //this is not a jump
    //     //     RT_Count = 0;
    //     //   }
    //     // }
    //     }
    // }
    if (!RT_collected){
        double d = (k_1 + 2*k_2 + 2*k_3 + k_4)/6.0;
        RT_sum += d;
        d_history.push_back(d);
        if (update_count >=RT_history){
            RT_sum -= d_history[0];
            d_history.erase(d_history.begin());
        }
        if (fabs(d) < RT_threshold){
            RT_Count++;
        }
        else{
            if (fabs(RT_sum/d_history.size()) > mean_threshold){ //mean is fluctuation
                RT_Count = 0;
            }
            else{
                RT_Count++;
            }
        }
    }



    //updating
    // t_prev = t;
    // x_prev = x;
    t += h;
    x += (k_1 + 2*k_2 + 2*k_3 + k_4)/6.0;

    //recording
    x_data.push_back(x);
    t_data.push_back(t);
}


void Neuron::updateRK4(double (Neuron::*f)(double), int c){

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
        double d = (k_1 + 2*k_2 + 2*k_3 + k_4)/6.0;
        RT_sum += d;
        d_history.push_back(d);
        if (update_count >=RT_history){
            RT_sum -= d_history[0];
            d_history.erase(d_history.begin());
        }
        if (fabs(d) < RT_threshold){
            RT_Count++;
        }
        else{
            if (fabs(RT_sum/d_history.size()) > mean_threshold){ //mean is fluctuation
                RT_Count = 0;
            }
            else{
                RT_Count++;
            }
        }
    }

    //updating
    // t_prev = t; // do this before t is updated to the next step.
    // x_prev = x;
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
        double d = (k_1 + 2*k_2 + 2*k_3 + k_4)/6.0;
        RT_sum += d;
        d_history.push_back(d);
        if (update_count >=RT_history){
            RT_sum -= d_history[0];
            d_history.erase(d_history.begin());
        }
        if (fabs(d) < RT_threshold){
            RT_Count++;
        }
        else{
            if (fabs(RT_sum/d_history.size()) > mean_threshold){ //mean is fluctuation
                RT_Count = 0;
            }
            else{
                RT_Count++;
            }
        }
    }

    //updating
    // t_prev = t;
    // x_prev = x;
    t += h;
    x += (k_1 + 2*k_2 + 2*k_3 + k_4)/6.0;

    //recording
    x_data.push_back(x);
    t_data.push_back(t);
}


void Neuron::updateEulerNoisy(double (Neuron::*f)(double)){
    update_count++;

    double RHS = computeRHS(this->t,this->x,(f));

    double noise = rand_normal(0,sqrt(h));

    t += h;
    x = x + RHS + noise;

    //recording
    x_data.push_back(x);
    t_data.push_back(t);
}


void Neuron::updateEulerNoisyIntegrateAll(double (Neuron::*f)(double)){
    update_count++;

    double RHS = computeRHSIntegrateAll(this->t,this->x,(f));

    double noise = rand_normal(0,sqrt(h));

    t += h;
    x = x + RHS + noise;

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

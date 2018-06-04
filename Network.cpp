#include "Network.h"

using namespace std;

Network::Network(int num_neurons, double w, double lambda, double time_step){
    this->num_neurons = num_neurons;

    time = 0;
    this->time_step = time_step;
    RT = 0;
    Acc = 0;

    for(int i=0;i<num_neurons;i++){
        Neuron* n = new Neuron(i,0,lambda,w,0.5,0,time_step);
        neuron_vector.push_back(n);
    }
}

Network::~Network(){
    for (int i=0;i<num_neurons;i++){
        delete neuron_vector[i];
    }
}

void Network::initialize(double diff){
    //Assume one winner, and diff is in [0,1]
    //Assume no biase in IC
    for (int i=0;i<num_neurons;i++){
        neuron_vector[i]->S = 1.0-diff;
        neuron_vector[i]->x = 0.5;
    }
    int winner_num = rand() % num_neurons;
    neuron_vector[winner_num]->S = 1.0;
    winners.push_back(neuron_vector[winner_num]);
}


void Network::update(double (Neuron::*f)(double)){
    time += time_step;
    bool RT_achieved = true;

    for (int i =0; i<num_neurons;i++){
        Neuron* toUpdate = neuron_vector[i];
        toUpdate->updateRK4(f);
        if (toUpdate->update_count > toUpdate->RT_history){
            if (toUpdate->RT_sum > toUpdate->RT_threshold){ //not yet RT
                RT_achieved = false;
            }
        }
        else{
            RT_achieved = false;
        }
    }

    if (RT==0){ //this part sets the RT should it be achieved.
        if (RT_achieved){
            for (int i=0;i<num_neurons;i++){
                Neuron* toUpdate = neuron_vector[i];
                toUpdate->RT_collected = true;
            }
            RT = time;
        }
    }

    //after updating everyone, renew the history
    for (int i=0;i<num_neurons;i++){
        Neuron* toUpdate = neuron_vector[i];
        toUpdate->t_prev = toUpdate->t;
        toUpdate->x_prev = toUpdate->x;
    }
}

void Network::updateIntegrateAll(double (Neuron::*f)(double)){
    time += time_step;
    bool RT_achieved = true;

    for (int i =0; i<num_neurons;i++){
        Neuron* toUpdate = neuron_vector[i];
        toUpdate->updateRK4IntegrateAll(f);
        if (toUpdate->update_count > toUpdate->RT_history){
            if (toUpdate->RT_sum > toUpdate->RT_threshold){ //not yet RT
                RT_achieved = false;
            }
        }
        else{
            RT_achieved = false;
        }
    }

    if (RT==0){ //this part sets the RT should it be achieved.
        if (RT_achieved){
            for (int i=0;i<num_neurons;i++){
                Neuron* toUpdate = neuron_vector[i];
                toUpdate->RT_collected = true;
            }
            RT = time;
        }
    }

    //after updating everyone, renew the history
    for (int i=0;i<num_neurons;i++){
        Neuron* toUpdate = neuron_vector[i];
        toUpdate->t_prev = toUpdate->t;
        toUpdate->x_prev = toUpdate->x;
    }
}

void Network::insertUndirectedConnection(Neuron* n1, Neuron* n2){
    if (contains(n1->neighbors,n2)){
        return; //check if the connection is already there.
    }
    n1->neighbors.push_back(n2);
    n2->neighbors.push_back(n1);
}

void Network::insertUndirectedConnection(int n1, int n2){
    if (n1 == n2){
        return;
    }
    Neuron* N1 = neuron_vector[n1];
    Neuron* N2 = neuron_vector[n2];
    insertUndirectedConnection(N1,N2);
}


void Network::insertUndirectedConnectionNoChecking(Neuron* n1, Neuron* n2){
    if (n1 ==n2){
        return;
    }
    n1->neighbors.push_back(n2);
    n2->neighbors.push_back(n1);
}

void Network::insertUndirectedConnectionNoChecking(int n1, int n2){
    Neuron* N1 = neuron_vector[n1];
    Neuron* N2 = neuron_vector[n2];
    insertUndirectedConnectionNoChecking(N1,N2);
}

void Network::removeUndirectedConnection(Neuron* n1, Neuron* n2){
    for (int i=0;i<n1->neighbors.size();i++){
        Neuron* current = n1->neighbors[i];
        if (current == n2){
            n1->neighbors.erase(n1->neighbors.begin()+i);
        }
    }
    for (int i=0;i<n2->neighbors.size();i++){
        Neuron* current = n2->neighbors[i];
        if (current == n1){
            n2->neighbors.erase(n2->neighbors.begin()+i);
        }
    }
}

void Network::removeUndirectedConnection(int n1, int n2){
    Neuron* N1 = neuron_vector[n1];
    Neuron* N2 = neuron_vector[n2];
    removeUndirectedConnection(N1,N2);
}

vector<Neuron*> Network::getNeighbors(Neuron* n){
    return n->neighbors;
}


void Network::constructRegularNetwork(int k){
    //follows the algorithm on constructing a ring lattice.
    //assumes k to be even
    k = k/2;

    for (int i=0;i<num_neurons;i++){
        Neuron* n = neuron_vector[i];
        for (int j=-k;j<=k;j++){
            if (j!=0){
                int toConnect = mod(i-j,num_neurons);
                insertUndirectedConnection(i,toConnect);
            }
        }
    }
}


// void Network::constructRandomNetwork(int k){
//     double thresholdProb = ((double)k)/num_neurons;
//     for (int i=0;i<num_neurons;i++){
//         Neuron* n = neuron_vector[i];
//         for (int j=0;j<num_neurons;j++){
//             //r is a random float in [0,1]
//             float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
//             if (r >= thresholdProb){
//                 insertUndirectedConnection(i,j);
//             }
//         }
//     }
// }

void Network::constructRandomNetwork(int p){
    for (int i=0;i<num_neurons;i++){
        for (int j=i+1;j<num_neurons;j++){
            //r is a random float in [0,1]
            float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            if (r <= p){
                insertUndirectedConnection(i,j);
            }
        }
    }
}

void Network::constructSmallWorldNetwork(int k, double p){
    constructRegularNetwork(k);
    //rewire
    
}

void Network::constructAllToAllNetwork(){
    for (int i=0;i<neuron_vector.size();i++){
        for (int j=i+1;j<num_neurons;j++){
            insertUndirectedConnectionNoChecking(i,j); //fine here since we're progressing linearly
        }
    }
}

void Network::computeAccuracy(){
    double max_winner_acc =0;
    for (int i=0;i<winners.size();i++){
        if (winners[i]->x > max_winner_acc){
            max_winner_acc = winners[i]->x;
        }
    }

    double max_loser_acc = 0;
    for (int i=0;i<neuron_vector.size();i++){
        Neuron* loser = neuron_vector[i];
        if (!contains(winners,loser) && loser->x > max_loser_acc){
            max_loser_acc = loser->x;
        }
    }

    Acc = max_winner_acc - max_loser_acc;
}
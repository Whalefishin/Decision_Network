#include <string>
#include <iostream>
#include <utility>
#include <ctime>

#include "Network.h"

using namespace std;

int main(){
    srand(6);

    Network* network = new Network(3,5,1,0.01);

    Neuron* n1 = network->neuron_vector[0];
    Neuron* n2 = network->neuron_vector[1];
    Neuron* n3 = network->neuron_vector[2];
    // n1->x = 0.1;
    // n2->x = 0.1;
    // n1->S = 0.35; //so n1 should be the winner.
    // n2->S = 0.3;

    network->constructAllToAllNetwork();
    // for (int i =0;i<network->num_neurons;i++){
    //     network->insertUndirectedConnection(0,i);
    // }

    for (int i=0;i<network->num_neurons;i++){
        Neuron* n = network->neuron_vector[i];
        n->x = 0.5;
        n->S = 0.3;
    }

    //winner parameters
    n1->S = 0.5;

    network->winners.push_back(n1);

    int update_times = 50;
    for (int t=0;t<update_times;t++){
        network->updateIntegrateAll(&Neuron::sigmActiv);
        if (t == update_times-1){ //last loop, collect accuracy
            network->computeAccuracy();
        }
    }

    cout << "RT: " + to_string(network->RT) << endl;
    cout << "Accuracy: " + to_string(network->Acc) << endl;


    //scaling data
    vector<double> w_data;
    vector<double> RT_data;
    vector<double> Acc_data;

    //scaling
    int num_networks = 30;
    int update_times_scaling = 1000;
    for (int i=0;i<num_networks;i++){
        //initialize
        double w = (double)i;
        Network scaling_network(2,w,1,0.01);
        scaling_network.constructAllToAllNetwork();
        Neuron* n1 = scaling_network.neuron_vector[0];
        Neuron* n2 = scaling_network.neuron_vector[1];
        //Neuron* n3 = scaling_network.neuron_vector[2];
        n1->x = 0.5;
        n2->x = 0.5;
        //n3->x = 0.5;
        n1->S = 0.5; //so n1 should be the winner.
        n2->S = 0.3;
        //n3->S = 0.1;
        scaling_network.winners.push_back(n1);
        //simulate
        for (int t=0;t<update_times_scaling;t++){
            scaling_network.updateIntegrateAll(&Neuron::sigmActiv);
            if (t==update_times_scaling-1){
                scaling_network.computeAccuracy();
            }
        }
        //collect
        w_data.push_back(w);
        RT_data.push_back(scaling_network.RT);
        Acc_data.push_back(scaling_network.Acc);
    }


    //3D plots
    int num_outer_loop = 10;
    int num_inner_loop = 10;
    int update_times_3D = 10000;

    vector<double> w_Vector_3D;
    vector<double> N_Vector_3D;
    vector<double> Acc_Vector_3D;

    for (int i=1;i<=num_outer_loop;i++){
        double N = i*10;
        for (int j=1;j<=num_inner_loop;j++){
            double w = j-1;
            Network network_3D(N,w,1,0.01);
            network_3D.constructAllToAllNetwork();
            network_3D.initialize(0.5);
            for (int k = 0;k<update_times_3D;k++){
                network_3D.updateIntegrateAll(&Neuron::sigmActiv);
                if (k==update_times_3D-1){
                    network_3D.computeAccuracy();
                }
            }
            N_Vector_3D.push_back(N);
            w_Vector_3D.push_back(w);
            Acc_Vector_3D.push_back(network_3D.Acc);
        }
    }


    //Outputting
    ofstream timeData("Data/Time.txt");
    ofstream x1Data("Data/x1.txt");
    ofstream x2Data("Data/x2.txt");
    ofstream singleNetworkParameters("Data/SingleNetworkParameters.txt");
    
    ofstream wData("Data/w.txt");
    ofstream RTData("Data/RT.txt");
    ofstream AccData("Data/Acc.txt");

    ofstream w_Data_3D("Data/3D_w.txt");
    ofstream N_Data_3D("Data/3D_N.txt");
    ofstream Acc_Data_3D("Data/3D_Acc.txt");


    for (int i=0;i<n1->t_data.size();i++){
        timeData << n1->t_data[i] << endl;
        x1Data << n1->x_data[i] << endl;
        x2Data << n2->x_data[i] << endl;
    }

    for (int i=0;i<num_networks;i++){
        wData << w_data[i] << endl;
        RTData << RT_data[i] << endl;
        AccData << Acc_data[i] << endl;
    }

    for (int i=0;i<N_Vector_3D.size();i++){
        w_Data_3D << w_Vector_3D[i] << endl;
        N_Data_3D << N_Vector_3D[i] << endl;
        Acc_Data_3D << Acc_Vector_3D[i] << endl;
    }

    //single network
    singleNetworkParameters << "Number of neurons: " + to_string(network->num_neurons) << endl;
    singleNetworkParameters << "Reaction Time: " + to_string(network->RT) << endl;
    singleNetworkParameters << "Accuracy: " + to_string(network->Acc) << endl;


    return 0;
}
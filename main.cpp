#include <string>
#include <iostream>
#include <utility>
#include <ctime>

#include "Network.h"

using namespace std;

int main(){
    //fixed random seed for consistency
    srand(6);

    int single_Network_neurons = 1000;
    Network* network = new Network(single_Network_neurons,5,1,0.01);

    Neuron* n1 = network->neuron_vector[0];
    Neuron* n2 = network->neuron_vector[1];
    Neuron* n3 = network->neuron_vector[2];
    // n1->x = 0.1;
    // n2->x = 0.1;
    // n1->S = 0.35; //so n1 should be the winner.
    // n2->S = 0.3;

    //network->constructRandomNetwork(0.5);
    //network->constructAllToAllNetwork();
    network->constructSmallWorldNetwork(single_Network_neurons/10,0.1);
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

    int** J = network->outputAdjacencyMtx();

    cout << "RT: " + to_string(network->RT) << endl;
    cout << "Accuracy: " + to_string(network->Acc) << endl;


    //scaling data
    vector<double> w_data;
    vector<double> RT_data;
    vector<double> Acc_data;

    //scaling
    int num_networks = 10;
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
    int num_outer_loop = 3;
    int num_inner_loop = 21;
    int update_times_3D = 10000;

    vector<double> w_Vector_3D;
    vector<double> N_Vector_3D;
    vector<double> diff_Vector_3D;
    vector<double> Acc_Vector_3D;
    vector<double> RT_Vector_3D;

    for (int i=1;i<=num_outer_loop;i++){
      cout << "big loop: " + to_string(i) << endl;
        //double N = 10;
        double N = i*10;
        //double diff = (i-1)*0.1;
        double diff = 0.8;
        for (int j=1;j<=num_inner_loop;j++){
            //double w = 1.0;
            double w = (double)(j-1);
            //double diff = (j-1)*0.1;
            Network network_3D(N,w/(N-1),1,0.01);
            network_3D.constructAllToAllNetwork();
            network_3D.initializeFairIC(1.0,diff);
            //network_3D.initialize(0.5);
            for (int k = 0;k<update_times_3D;k++){
                network_3D.updateIntegrateAll(&Neuron::sigmActiv);
                if (k==update_times_3D-1){
                    network_3D.computeAccuracy();
                }
            }
            N_Vector_3D.push_back(N);
            w_Vector_3D.push_back(w);
            diff_Vector_3D.push_back(diff);
            Acc_Vector_3D.push_back(network_3D.Acc);
            RT_Vector_3D.push_back(network_3D.RT);
        }
    }


    //3D with averaged fair IC
    int num_outer_loop_AVG_Fair = 10;
    int num_inner_loop_AVG_Fair = 21;
    int update_times_3D_AVG_Fair = 5000;
    int num_IC = 10; //number of IC to avg.

    vector<double> w_Vector_3D_AVG_Fair;
    vector<double> N_Vector_3D_AVG_Fair;
    vector<double> diff_Vector_3D_AVG_Fair;
    vector<double> Acc_Vector_3D_AVG_Fair;
    vector<double> RT_Vector_3D_AVG_Fair;


    for (int i=1;i<=num_outer_loop_AVG_Fair;i++){
        //double N = 10;
        double N = i*10;
        //double diff = (i-1)*0.1;
        double diff = 0.5;
        for (int j=1;j<=num_inner_loop_AVG_Fair;j++){
            double Acc_sum =0; //IC avg.
            double RT_sum = 0; //IC avg.
            double w = (double)(j-1);
            for (int l=1;l<=num_IC;l++){
                double IC = l*0.09;
                //double w = 1.0;
                //double diff = (j-1)*0.1;
                Network network_3D(N,w/(N-1),1,0.01);
                network_3D.constructAllToAllNetwork();
                network_3D.initializeFairIC(IC,diff);
                //network_3D.initialize(0.5);
                for (int k = 0;k<update_times_3D_AVG_Fair;k++){
                    network_3D.updateIntegrateAll(&Neuron::sigmActiv);
                    if (k==update_times_3D_AVG_Fair-1){
                        network_3D.computeAccuracy();
                    }
                }
                Acc_sum += max(0,network_3D.Acc); //if Acc is neg, set to 0.
                RT_sum += network_3D.RT;
            }
            N_Vector_3D_AVG_Fair.push_back(N);
            w_Vector_3D_AVG_Fair.push_back(w);
            diff_Vector_3D_AVG_Fair.push_back(diff);
            Acc_Vector_3D_AVG_Fair.push_back(Acc_sum/num_IC);
            RT_Vector_3D_AVG_Fair.push_back(RT_sum/num_IC);
        }
    }


    //Outputting
    ofstream timeData("Data/Time.txt");
    ofstream x1Data("Data/x1.txt");
    ofstream x2Data("Data/x2.txt");
    ofstream JmtxData("Data/J.txt");
    ofstream singleNetworkParameters("Data/SingleNetworkParameters.txt");

    ofstream wData("Data/w.txt");
    ofstream RTData("Data/RT.txt");
    ofstream AccData("Data/Acc.txt");

    ofstream w_Data_3D("Data/3D_w.txt");
    ofstream N_Data_3D("Data/3D_N.txt");
    ofstream diff_Data_3D("Data/3D_diff.txt");
    ofstream Acc_Data_3D("Data/3D_Acc.txt");
    ofstream RT_Data_3D("Data/3D_RT.txt");

    ofstream w_Data_3D_AVG_Fair("Data/3D_w_AVG_Fair.txt");
    ofstream N_Data_3D_AVG_Fair("Data/3D_N_AVG_Fair.txt");
    ofstream diff_Data_3D_AVG_Fair("Data/3D_diff_AVG_Fair.txt");
    ofstream Acc_Data_3D_AVG_Fair("Data/3D_Acc_AVG_Fair.txt");
    ofstream RT_Data_3D_AVG_Fair("Data/3D_RT_AVG_Fair.txt");

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
        diff_Data_3D << diff_Vector_3D[i] << endl;
        Acc_Data_3D << Acc_Vector_3D[i] << endl;
        RT_Data_3D << RT_Vector_3D[i] << endl;
    }

    for (int i=0;i<N_Vector_3D_AVG_Fair.size();i++){
        w_Data_3D_AVG_Fair << w_Vector_3D_AVG_Fair[i] << endl;
        N_Data_3D_AVG_Fair << N_Vector_3D_AVG_Fair[i] << endl;
        diff_Data_3D_AVG_Fair << diff_Vector_3D_AVG_Fair[i] << endl;
        Acc_Data_3D_AVG_Fair << Acc_Vector_3D_AVG_Fair[i] << endl;
        RT_Data_3D_AVG_Fair << RT_Vector_3D_AVG_Fair[i] << endl;
    }

    //single network
    singleNetworkParameters << "Number of neurons: " + to_string(network->num_neurons) << endl;
    singleNetworkParameters << "Reaction Time: " + to_string(network->RT) << endl;
    singleNetworkParameters << "Accuracy: " + to_string(network->Acc) << endl;
    //J mtx
    for (int i=0;i<single_Network_neurons;i++){
        for (int j=0;j<single_Network_neurons;j++){
            JmtxData << to_string(J[i][j]) + " ";
        }
        JmtxData << endl;
    }

    for (int i=0;i<single_Network_neurons;i++){
        delete[] J[i];
    }
    delete[] J;

    delete network;

    return 0;
}

#include <string>
#include <iostream>
#include <utility>
#include <ctime>
#include <thread>
#include <mutex>


#include "Network.h"

using namespace std;

mutex mtx;

struct Data{
    double N;
    double diff;
    double p;
    int k;
    int update_times;
    int num_inner_loop_ult;
    int diff_count;
    int gain_type;
    int sep_gain;
    int normalization;
    int biased_IC;
    vector<double> diff_vector;
    vector<double> neuron_vector;
    vector<int> IC_vector;
};

//WorkLoop for W vs. N
void workLoop(Data custom_data, vector<vector<double> >& W_N_N_Vector,
    vector<vector<double> >& W_N_W_Vector,
    vector<vector<double> >& W_N_diff_Vector,
    vector<vector<double> >& W_N_Acc_Vector,
    vector<vector<double> >& W_N_AccMean_Vector,
    vector<vector<double> >& W_N_RT_Vector,
    vector<vector<double> >& W_N_Acc_Var_Vector,
    vector<vector<double> >& W_N_RT_Var_Vector
  ){

    for (int j=1;j<=custom_data.num_inner_loop_ult;j++){
        double W = 0;
        if (j <=4){
          W = (j-1)*0.25;
        }
        else{
          W = (double)(j-4);
        }
        int index = 0;
        for (int d=0;d<custom_data.diff_count;d++){
            double diff_ult = custom_data.diff_vector[d];
            for (int g=0;g<custom_data.gain_type;g++){
                for (int s=0;s<custom_data.sep_gain;s++){
                    for (int n=0;n<custom_data.normalization;n++){
                        for (int b=0;b<custom_data.biased_IC;b++){
                            for (int dist=0;dist<2;dist++){
                                int num_IC = custom_data.IC_vector[b];                                //double w = 1.0;
                                double Acc_sum =0; //IC avg.
                                double Acc_mean = 0;
                                double RT_sum = 0; //IC avg.
                                vector<double> variance_Acc;
                                vector<double> variance_RT;
                                for (int l=1;l<=num_IC;l++){
                                    double IC = 0.1 + 0.8/num_IC * l;
                                    Network network_3D(custom_data.N,W,1,0.1,n);
                                    network_3D.constructAllToAllNetwork();
                                    network_3D.initializeWithChoice(dist,b,IC,diff_ult);
                                    for (int k = 0;k<custom_data.update_times;k++){
                                        network_3D.updateWithChoice(s,g);
                                        if (k==custom_data.update_times-1){
                                            network_3D.computeAccuracy();
                                        }
                                    }
                                    variance_Acc.push_back(network_3D.Acc);
                                    variance_RT.push_back(network_3D.RT);
                                    Acc_sum += max(0,network_3D.Acc); //if Acc is neg, set to 0.
                                    Acc_mean += max(0,network_3D.Acc_mean);
                                    RT_sum += network_3D.RT;
                                }
                                //data collection
                                //int index = b*1 + n*2 + s*4 + g*8;
                                mtx.lock();
                                W_N_N_Vector[index].push_back(custom_data.N);
                                W_N_W_Vector[index].push_back(W);
                                W_N_diff_Vector[index].push_back(diff_ult);
                                W_N_Acc_Vector[index].push_back(Acc_sum/num_IC);
                                W_N_AccMean_Vector[index].push_back(Acc_mean/num_IC);
                                W_N_RT_Vector[index].push_back(RT_sum/num_IC);
                                W_N_Acc_Var_Vector[index].push_back(standardDeviation(variance_Acc));
                                W_N_RT_Var_Vector[index].push_back(standardDeviation(variance_RT));
                                mtx.unlock();
                        //update index out of the simulation loop
                                index++;
                            }
                        }
                    }
                }
            }
        }
    }
}

//Work loop for W vs. Diff
void workLoop_2(Data custom_data, vector<vector<double> >& W_Diff_N_Vector,
    vector<vector<double> >& W_Diff_W_Vector,
    vector<vector<double> >& W_Diff_diff_Vector,
    vector<vector<double> >& W_Diff_Acc_Vector,
    vector<vector<double> >& W_Diff_AccMean_Vector,
    vector<vector<double> >& W_Diff_RT_Vector,
    vector<vector<double> >& W_Diff_Acc_Var_Vector,
    vector<vector<double> >& W_Diff_RT_Var_Vector){

    for (int j=1;j<=custom_data.num_inner_loop_ult;j++){
        double W = 0;
        if (j <=4){
          W = (j-1)*0.25;
        }
        else{
          W = (double)(j-4);
        }
        int index = 0;
        for (int d=0;d<custom_data.diff_count;d++){
            //double diff_ult = custom_data.diff_vector[d];
            double neuron_count = custom_data.neuron_vector[d];
            for (int g=0;g<custom_data.gain_type;g++){
                for (int s=0;s<custom_data.sep_gain;s++){
                    for (int n=0;n<custom_data.normalization;n++){
                        for (int b=0;b<custom_data.biased_IC;b++){
                            for (int dist=0; dist<2;dist++){
                                int num_IC = custom_data.IC_vector[b];                                //double w = 1.0;
                                double Acc_sum =0; //IC avg.
                                double Acc_mean = 0;
                                double RT_sum = 0; //IC avg.
                                vector<double> variance_Acc;
                                vector<double> variance_RT;
                                for (int l=1;l<=num_IC;l++){
                                    double IC = 0.1 + 0.8/num_IC * l;
                                    Network network_3D(neuron_count,W,1,0.1,n);
                                    network_3D.constructAllToAllNetwork();
                                    network_3D.initializeWithChoice(dist,b,IC,custom_data.diff);
                                    for (int k = 0;k<custom_data.update_times;k++){
                                        network_3D.updateWithChoice(s,g);
                                        if (k==custom_data.update_times-1){
                                            network_3D.computeAccuracy();
                                        }
                                    }
                                    variance_Acc.push_back(network_3D.Acc);
                                    variance_RT.push_back(network_3D.RT);
                                    Acc_sum += max(0,network_3D.Acc); //if Acc is neg, set to 0.
                                    Acc_mean += max(0,network_3D.Acc_mean);
                                    RT_sum += network_3D.RT;
                                }
                                //data collection
                                //int index = b*1 + n*2 + s*4 + g*8;
                                mtx.lock();
                                W_Diff_N_Vector[index].push_back(neuron_count);
                                W_Diff_W_Vector[index].push_back(W);
                                W_Diff_diff_Vector[index].push_back(custom_data.diff);
                                W_Diff_Acc_Vector[index].push_back(Acc_sum/num_IC);
                                W_Diff_AccMean_Vector[index].push_back(Acc_mean/num_IC);
                                W_Diff_RT_Vector[index].push_back(RT_sum/num_IC);
                                W_Diff_Acc_Var_Vector[index].push_back(standardDeviation(variance_Acc));
                                W_Diff_RT_Var_Vector[index].push_back(standardDeviation(variance_RT));
                                mtx.unlock();
                        //update index out of the simulation loop
                                index++;
                            }
                        }
                    }
                }
            }
        }
    }
}

// W vs. P
void workLoop_3(Data custom_data, vector<vector<double> >& W_P_N_Vector,
  vector<vector<double> >& W_P_P_Vector,
    vector<vector<double> >& W_P_W_Vector,
    vector<vector<double> >& W_P_diff_Vector,
    vector<vector<double> >& W_P_Acc_Vector,
    vector<vector<double> >& W_P_AccMean_Vector,
    vector<vector<double> >& W_P_RT_Vector,
    vector<vector<double> >& W_P_Acc_Var_Vector,
        vector<vector<double> >& W_P_RT_Var_Vector){

    int k =30;

    for (int j=1;j<=custom_data.num_inner_loop_ult;j++){
        double W = 0;
        if (j <=4){
          W = (j-1)*0.25;
        }
        else{
          W = (double)(j-4);
        }
        int index = 0;
        for (int d=0;d<custom_data.diff_count;d++){
            double diff_ult = custom_data.diff_vector[d];
            for (int g=0;g<custom_data.gain_type;g++){
                for (int s=0;s<custom_data.sep_gain;s++){
                    for (int n=0;n<custom_data.normalization;n++){
                        for (int b=0;b<custom_data.biased_IC;b++){
                            for (int dist=0;dist<2;dist++){
                                int num_IC = custom_data.IC_vector[b];
                                double Acc_sum =0; //IC avg.
                                double Acc_Mean = 0;
                                double RT_sum = 0; //IC avg.
                                vector<double> variance_Acc;
                                vector<double> variance_RT;
                                for (int l=1;l<=num_IC;l++){
                                    double IC = 0.1 + 0.8/num_IC * l;
                                    Network network_3D(custom_data.N,W,1,0.01,n,k);
                                    //network_3D.constructAllToAllNetwork();
                                    network_3D.constructSmallWorldNetwork(k,custom_data.p);
                                    network_3D.initializeWithChoice(dist,b,IC,diff_ult);
                                    for (int k = 0;k<custom_data.update_times;k++){
                                        network_3D.updateWithChoice(s,g);
                                        if (k==custom_data.update_times-1){
                                            network_3D.computeAccuracy();
                                        }
                                    }
                                    variance_Acc.push_back(network_3D.Acc);
                                    variance_RT.push_back(network_3D.RT);
                                    Acc_sum += max(0,network_3D.Acc); //if Acc is neg, set to 0.
                                    Acc_Mean += max(0,network_3D.Acc_mean);
                                    RT_sum += network_3D.RT;
                                }
                                //data collection
                                //int index = b*1 + n*2 + s*4 + g*8;
                                mtx.lock();
                                W_P_N_Vector[index].push_back(custom_data.N);
                                W_P_P_Vector[index].push_back(custom_data.p);
                                W_P_W_Vector[index].push_back(W);
                                W_P_diff_Vector[index].push_back(diff_ult);
                                W_P_Acc_Vector[index].push_back(Acc_sum/num_IC);
                                W_P_AccMean_Vector[index].push_back(Acc_Mean/num_IC);
                                W_P_RT_Vector[index].push_back(RT_sum/num_IC);
                                W_P_Acc_Var_Vector[index].push_back(standardDeviation(variance_Acc));
                                W_P_RT_Var_Vector[index].push_back(standardDeviation(variance_RT));
                                mtx.unlock();
                        //update index out of the simulation loop
                                index++;
                            }
                        }
                    }
                }
            }
        }
    }
}

// Regular to All-to-All W vs. K
void workLoop_4(Data custom_data, vector<vector<double> >& W_Regular_N_Vector,
  vector<vector<double> >& W_Regular_P_Vector,
  vector<vector<double> >& W_Regular_K_Vector,
    vector<vector<double> >& W_Regular_W_Vector,
    vector<vector<double> >& W_Regular_diff_Vector,
    vector<vector<double> >& W_Regular_Acc_Vector,
    vector<vector<double> >& W_Regular_AccMean_Vector,
    vector<vector<double> >& W_Regular_RT_Vector,
    vector<vector<double> >& W_Regular_Acc_Var_Vector,
        vector<vector<double> >& W_Regular_RT_Var_Vector){


    for (int j=1;j<=custom_data.num_inner_loop_ult;j++){
        double W = 0;
        if (j <=4){
          W = (j-1)*0.25;
        }
        else{
          W = (double)(j-4);
        }
        int index = 0;
        for (int d=0;d<custom_data.diff_count;d++){
            double diff_ult = custom_data.diff_vector[d];
            for (int g=0;g<custom_data.gain_type;g++){
                for (int s=0;s<custom_data.sep_gain;s++){
                    for (int n=0;n<custom_data.normalization;n++){
                        for (int b=0;b<custom_data.biased_IC;b++){
                            for (int dist=0;dist<2;dist++){
                                int num_IC = custom_data.IC_vector[b];
                                double Acc_sum =0; //IC avg.
                                double Acc_Mean = 0;
                                double RT_sum = 0; //IC avg.
                                vector<double> variance_Acc;
                                vector<double> variance_RT;
                                for (int l=1;l<=num_IC;l++){
                                    double IC = 0.1 + 0.8/num_IC * l;
                                    Network network_3D(custom_data.N,W,1,0.1,n,custom_data.k);
                                    //network_3D.constructAllToAllNetwork();
                                    network_3D.constructRegularNetwork(custom_data.k);
                                    network_3D.initializeWithChoice(dist,b,IC,diff_ult);
                                    for (int k = 0;k<custom_data.update_times;k++){
                                        network_3D.updateWithChoice(s,g);
                                        if (k==custom_data.update_times-1){
                                            network_3D.computeAccuracy();
                                        }
                                    }
                                    variance_Acc.push_back(network_3D.Acc);
                                    variance_RT.push_back(network_3D.RT);
                                    Acc_sum += max(0,network_3D.Acc); //if Acc is neg, set to 0.
                                    Acc_Mean += max(0,network_3D.Acc_mean);
                                    RT_sum += network_3D.RT;
                                }
                                //data collection
                                //int index = b*1 + n*2 + s*4 + g*8;
                                mtx.lock();
                                W_Regular_N_Vector[index].push_back(custom_data.N);
                                W_Regular_P_Vector[index].push_back(custom_data.p);
                                W_Regular_K_Vector[index].push_back(custom_data.k);
                                W_Regular_W_Vector[index].push_back(W);
                                W_Regular_diff_Vector[index].push_back(diff_ult);
                                W_Regular_Acc_Vector[index].push_back(Acc_sum/num_IC);
                                W_Regular_AccMean_Vector[index].push_back(Acc_Mean/num_IC);
                                W_Regular_RT_Vector[index].push_back(RT_sum/num_IC);
                                W_Regular_Acc_Var_Vector[index].push_back(standardDeviation(variance_Acc));
                                W_Regular_RT_Var_Vector[index].push_back(standardDeviation(variance_RT));
                                mtx.unlock();
                        //update index out of the simulation loop
                                index++;
                            }
                        }
                    }
                }
            }
        }
    }
}

// Random to All-to-All W vs. P
void workLoop_5(Data custom_data, vector<vector<double> >& W_Random_N_Vector,
  vector<vector<double> >& W_Random_P_Vector,
  vector<vector<double> >& W_Random_K_Vector,
    vector<vector<double> >& W_Random_W_Vector,
    vector<vector<double> >& W_Random_diff_Vector,
    vector<vector<double> >& W_Random_Acc_Vector,
    vector<vector<double> >& W_Random_AccMean_Vector,
    vector<vector<double> >& W_Random_RT_Vector,
    vector<vector<double> >& W_Random_Acc_Var_Vector,
        vector<vector<double> >& W_Random_RT_Var_Vector){


    for (int j=1;j<=custom_data.num_inner_loop_ult;j++){
        double W = 0;
        if (j <=4){
          W = (j-1)*0.25;
        }
        else{
          W = (double)(j-4);
        }
        int index = 0;
        for (int d=0;d<custom_data.diff_count;d++){
            double diff_ult = custom_data.diff_vector[d];
            for (int g=0;g<custom_data.gain_type;g++){
                for (int s=0;s<custom_data.sep_gain;s++){
                    for (int n=0;n<custom_data.normalization;n++){
                        for (int b=0;b<custom_data.biased_IC;b++){
                            for (int dist=0;dist<2;dist++){
                                int num_IC = custom_data.IC_vector[b];
                                double Acc_sum =0; //IC avg.
                                double Acc_mean =0;
                                double RT_sum = 0; //IC avg.
                                vector<double> variance_Acc;
                                vector<double> variance_RT;
                                for (int l=1;l<=num_IC;l++){
                                    double IC = 0.1 + 0.8/num_IC * l;
                                    Network network_3D(custom_data.N,W,1,0.1,n,custom_data.p*custom_data.N);
                                    //network_3D.constructAllToAllNetwork();
                                    network_3D.constructRandomNetwork(custom_data.p);
                                    network_3D.initializeWithChoice(dist,b,IC,diff_ult);
                                    for (int k = 0;k<custom_data.update_times;k++){
                                        network_3D.updateWithChoice(s,g);
                                        if (k==custom_data.update_times-1){
                                            network_3D.computeAccuracy();
                                        }
                                    }
                                    variance_Acc.push_back(network_3D.Acc);
                                    variance_RT.push_back(network_3D.RT);
                                    Acc_sum += max(0,network_3D.Acc); //if Acc is neg, set to 0.
                                    Acc_mean += max(0,network_3D.Acc_mean);
                                    RT_sum += network_3D.RT;
                                }
                                //data collection
                                //int index = b*1 + n*2 + s*4 + g*8;
                                mtx.lock();
                                W_Random_N_Vector[index].push_back(custom_data.N);
                                W_Random_P_Vector[index].push_back(custom_data.p);
                                W_Random_K_Vector[index].push_back(custom_data.k);
                                W_Random_W_Vector[index].push_back(W);
                                W_Random_diff_Vector[index].push_back(diff_ult);
                                W_Random_Acc_Vector[index].push_back(Acc_sum/num_IC);
                                W_Random_AccMean_Vector[index].push_back(Acc_mean/num_IC);
                                W_Random_RT_Vector[index].push_back(RT_sum/num_IC);
                                W_Random_Acc_Var_Vector[index].push_back(standardDeviation(variance_Acc));
                                W_Random_RT_Var_Vector[index].push_back(standardDeviation(variance_RT));
                                mtx.unlock();
                        //update index out of the simulation loop
                                index++;
                            }
                        }
                    }
                }
            }
        }
    }
}


int main(){


    //Controls which runs we want to do.

    bool run_W_N = false;
    bool run_W_Diff = false;
    bool run_W_P = false;
    bool run_W_Regular = false;
    bool run_W_Random = false;



    //fixed random seed for consistency
    srand(6);

    int single_Network_neurons = 50;
    Network* network = new Network(single_Network_neurons,10,1,0.1,1);

    Neuron* n1 = network->neuron_vector[0];
    Neuron* n2 = network->neuron_vector[1];
    Neuron* n3 = network->neuron_vector[2];
    Neuron* n4 = network->neuron_vector[3];
    Neuron* n5 = network->neuron_vector[4];
    // n1->x = 0.1;
    // n2->x = 0.1;
    // n1->S = 0.35; //so n1 should be the winner.
    // n2->S = 0.3;

    //network->constructRandomNetwork(0.5);
    network->constructAllToAllNetwork();
    //network->constructSmallWorldNetwork(single_Network_neurons/10,0.1);
    //network->initializeWithChoice(0,0.5,0.5);

    // for (int i =0;i<network->num_neurons;i++){
    //     network->insertUndirectedConnection(0,i);
    // }

    for (int i=0;i<network->num_neurons;i++){
        Neuron* n = network->neuron_vector[i];
        //n->x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        //cout << n->x << endl;
        n->x = 0.5;
        n->x_prev = n->x;
        n->S = 0.8;
    }

    //winner parameters
    n1->S = 1.0;
    network->winners.push_back(n1);

    int update_times = 500;
    for (int t=0;t<update_times;t++){
        network->updateIntegrateAll(&Neuron::binaryActiv);
        if (t == update_times-1){ //last loop, collect accuracy
            network->computeAccuracy();
        }
    }

    double xData = -1.0;
    int index_w = 0;

    for (int i=0;i<single_Network_neurons;i++){
        Neuron* n = network->neuron_vector[i];
        if (n->x > xData){
            xData = n->x;
            index_w = i;
        }
    }

    cout << "winner index: " + to_string(index_w) << endl;

    Neuron* n_winner = network->neuron_vector[index_w];



    int** J = network->outputAdjacencyMtx();

    cout << "RT: " + to_string(network->RT) << endl;
    cout << "Accuracy: " + to_string(network->Acc) << endl;


    //scaling data
    vector<double> w_data;
    vector<double> RT_data;
    vector<double> Acc_data;

    //scaling
    int num_networks = 10;
    int update_times_scaling = 0;
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
    int num_inner_loop = 11;
    int update_times_3D = 0;

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
        double diff = 0.2;
        for (int j=1;j<=num_inner_loop;j++){
            //double w = 1.0;
            double w = (double)(j-1);
            //double diff = (j-1)*0.1;
            Network network_3D(N,w/(N-1),1,0.1);
            network_3D.constructAllToAllNetwork();
            network_3D.initializeFairICNoDist(1.0,diff);
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
    int num_inner_loop_AVG_Fair = 11;
    int update_times_3D_AVG_Fair = 0;
    int num_IC_AVG_Fair = 10; //number of IC to avg.

    vector<double> w_Vector_3D_AVG_Fair;
    vector<double> N_Vector_3D_AVG_Fair;
    vector<double> diff_Vector_3D_AVG_Fair;
    vector<double> Acc_Vector_3D_AVG_Fair;
    vector<double> RT_Vector_3D_AVG_Fair;


    for (int i=1;i<=num_outer_loop_AVG_Fair;i++){
        //double N = 10;
        double N = i*10;
        //double diff = (i-1)*0.1;
        double diff = 0.2;
        for (int j=1;j<=num_inner_loop_AVG_Fair;j++){
            double Acc_sum =0; //IC avg.
            double RT_sum = 0; //IC avg.
            double w = (double)(j-1);
            for (int l=1;l<=num_IC_AVG_Fair;l++){
                double IC = 0.1 + 0.8/num_IC_AVG_Fair * l;
                //double IC = l*0.09;
                //double w = 1.0;
                //double diff = (j-1)*0.1;
                Network network_3D(N,w/(N-1),1,0.1);
                network_3D.constructAllToAllNetwork();
                network_3D.initializeFairICNoDist(IC,diff);
                //network_3D.initialize(0.5);
                for (int k = 0;k<update_times_3D_AVG_Fair;k++){
                    network_3D.update(&Neuron::sigmActiv);
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
            Acc_Vector_3D_AVG_Fair.push_back(Acc_sum/num_IC_AVG_Fair);
            RT_Vector_3D_AVG_Fair.push_back(RT_sum/num_IC_AVG_Fair);
        }
    }

    //cout << "before ult" << endl;


    //Behold, the ultimate version for W vs. N
    int gain_type = 2;
    int sep_gain = 2;
    int normalization = 2;
    int biased_IC = 2;
    int diff_count = 3;
    int distributed_input = 2;
    int index = 0;

    int num_outer_loop_ult = 10;
    int num_inner_loop_ult = 14;
    int update_times_ult = 1000;
    //double diff_ult = 0.5;
    vector<double> diff_vector;
    diff_vector.push_back(0.2);
    diff_vector.push_back(0.5);
    diff_vector.push_back(0.8);

    int num_neuron_count = 3;
    vector<double> num_neuron_vector;
    num_neuron_vector.push_back(10);
    num_neuron_vector.push_back(50);
    num_neuron_vector.push_back(100);

    //Writing W vs N data
    vector<string> diff_names;
    vector<string> gain_names;
    vector<string> sep_names;
    vector<string> norm_names;
    vector<string> biased_names;
    vector<string> dist_names;
    diff_names.push_back("_Easy");
    diff_names.push_back("_Medium");
    diff_names.push_back("_Hard");
    gain_names.push_back("_Sigm");
    gain_names.push_back("_Binary");
    sep_names.push_back("_IntAll");
    sep_names.push_back("_Sep");
    norm_names.push_back("_YesNorm");
    norm_names.push_back("_NoNorm");
    biased_names.push_back("_Fair");
    biased_names.push_back("_Unfair");
    dist_names.push_back("_NoDist");
    dist_names.push_back("_YesDist");

    string W_N_base_name = "W_N";
    string base_name = "W_Diff";

    if (run_W_N){
        int num_Fair_IC_ult = 10;
        int num_Unfair_IC_ult = 100;
        vector<int> IC_vector;
        IC_vector.push_back(num_Fair_IC_ult);
        IC_vector.push_back(num_Unfair_IC_ult);

        vector<vector<double> > W_N_N_Vector;
        vector<vector<double> > W_N_W_Vector;
        vector<vector<double> > W_N_diff_Vector;
        vector<vector<double> > W_N_Acc_Vector;
        vector<vector<double> > W_N_AccMean_Vector;
        vector<vector<double> > W_N_RT_Vector;
        vector<vector<double> > W_N_Acc_Var_Vector;
        vector<vector<double> > W_N_RT_Var_Vector;

        for (int i=0;i<diff_count*gain_type*sep_gain*normalization*biased_IC*distributed_input;i++){
            vector<double> toPush;
            W_N_N_Vector.push_back(toPush);
            W_N_W_Vector.push_back(toPush);
            W_N_diff_Vector.push_back(toPush);
            W_N_Acc_Vector.push_back(toPush);
            W_N_AccMean_Vector.push_back(toPush);
            W_N_RT_Vector.push_back(toPush);
            W_N_Acc_Var_Vector.push_back(toPush);
            W_N_RT_Var_Vector.push_back(toPush);
        }



    //Parallel version
        vector<thread> threads;

        Data computing_data;

        for (int i=1;i<=num_outer_loop_ult;i++){
            double N = i*10;
            computing_data.N = N;
            computing_data.diff = 0;
            computing_data.p = 0;
            computing_data.k = 0;
            computing_data.update_times = update_times_ult;
            computing_data.num_inner_loop_ult = num_inner_loop_ult;
            computing_data.diff_count = diff_count;
            computing_data.gain_type = gain_type;
            computing_data.sep_gain = sep_gain;
            computing_data.normalization = normalization;
            computing_data.biased_IC = biased_IC;
            computing_data.diff_vector = diff_vector;
            computing_data.IC_vector = IC_vector;

            threads.push_back(thread(workLoop,computing_data,ref(W_N_N_Vector),ref(W_N_W_Vector),
            ref(W_N_diff_Vector),ref(W_N_Acc_Vector),ref(W_N_AccMean_Vector),ref(W_N_RT_Vector), ref(W_N_Acc_Var_Vector),
        ref(W_N_RT_Var_Vector)));
        }

        for (int i=0;i<threads.size();i++){
            threads[i].join();
        }


        //outputting data
        index=0;

        for (int d=0;d<diff_count;d++){
            for (int g=0;g<gain_type;g++){
                for (int s=0;s<sep_gain;s++){
                    for (int n=0;n<normalization;n++){
                        for (int b=0;b<biased_IC;b++){
                            for (int dist=0;dist<2;dist++){
                                string filename1 = "Data/" + W_N_base_name +"_N"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b]+ dist_names[dist] + diff_names[d] +  ".txt";
                                string filename2 = "Data/" +W_N_base_name +"_W"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b]+dist_names[dist] + diff_names[d] +  ".txt";
                                string filename3 = "Data/" +W_N_base_name +"_Diff"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b]+dist_names[dist] + diff_names[d] +  ".txt";
                                string filename4 = "Data/" +W_N_base_name +"_Acc"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b]+dist_names[dist] + diff_names[d] +  ".txt";
                                string filename5 = "Data/" +W_N_base_name +"_RT"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b] + dist_names[dist] + diff_names[d] + ".txt";
                                string filename6 = "Data/" +W_N_base_name +"_Acc_Var"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b] + dist_names[dist] + diff_names[d] + ".txt";
                                string filename7 = "Data/" +W_N_base_name +"_RT_Var"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b] + dist_names[dist] + diff_names[d] + ".txt";
                                string filename8 = "Data/" +W_N_base_name +"_AccMean"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b] + dist_names[dist] + diff_names[d] + ".txt";
                                ofstream ToWrite1(filename1);
                                ofstream ToWrite2(filename2);
                                ofstream ToWrite3(filename3);
                                ofstream ToWrite4(filename4);
                                ofstream ToWrite5(filename5);
                                ofstream ToWrite6(filename6);
                                ofstream ToWrite7(filename7);
                                ofstream ToWrite8(filename8);
                                //int index = b*1 + n*2 + s*4 + g*8;
                                for (int i=0;i<W_N_N_Vector[index].size();i++){
                                    ToWrite1 << W_N_N_Vector[index][i] << endl;
                                    ToWrite2 << W_N_W_Vector[index][i] << endl;
                                    ToWrite3 << W_N_diff_Vector[index][i] << endl;
                                    ToWrite4 << W_N_Acc_Vector[index][i] << endl;
                                    ToWrite5 << W_N_RT_Vector[index][i] << endl;
                                    ToWrite6 << W_N_Acc_Var_Vector[index][i] << endl;
                                    ToWrite7 << W_N_RT_Var_Vector[index][i] << endl;
                                    ToWrite8 << W_N_AccMean_Vector[index][i] << endl;
                                }
                                index++;
                            }
                        }
                    }
                }
            }
        }

    }





    //Behold again, the ultimate version for W vs. Diff
    if (run_W_Diff){
        int num_outer_loop_ult_2 = 10;
        int num_inner_loop_ult_2 = 14;
        int update_times_ult_2 = 1000;
        //double diff_ult = 0.5;

        int num_Fair_IC_ult_2 = 10;
        int num_Unfair_IC_ult_2 = 100;
        vector<int> IC_vector_2;
        IC_vector_2.push_back(num_Fair_IC_ult_2);
        IC_vector_2.push_back(num_Unfair_IC_ult_2);

        vector<vector<double> > W_Diff_N_Vector;
        vector<vector<double> > W_Diff_W_Vector;
        vector<vector<double> > W_Diff_diff_Vector;
        vector<vector<double> > W_Diff_Acc_Vector;
        vector<vector<double> > W_Diff_AccMean_Vector;
        vector<vector<double> > W_Diff_RT_Vector;
        vector<vector<double> > W_Diff_Acc_Var_Vector;
        vector<vector<double> > W_Diff_RT_Var_Vector;

        for (int i=0;i<num_neuron_count*gain_type*sep_gain*normalization*biased_IC*distributed_input;i++){
            vector<double> toPush;
            W_Diff_N_Vector.push_back(toPush);
            W_Diff_W_Vector.push_back(toPush);
            W_Diff_diff_Vector.push_back(toPush);
            W_Diff_Acc_Vector.push_back(toPush);
            W_Diff_AccMean_Vector.push_back(toPush);
            W_Diff_RT_Vector.push_back(toPush);
            W_Diff_Acc_Var_Vector.push_back(toPush);
            W_Diff_RT_Var_Vector.push_back(toPush);
        }

        index = 0;
        vector<thread> threads_2;

        Data computing_data_2;

        for (int i=1;i<=num_outer_loop_ult_2;i++){
            double diff = 0.1 + 0.8/num_outer_loop_ult_2 * i;
            computing_data_2.N = 0;
            computing_data_2.diff = diff;
            computing_data_2.p = 0;
            computing_data_2.k = 0;
            computing_data_2.update_times = update_times_ult_2;
            computing_data_2.num_inner_loop_ult = num_inner_loop_ult_2;
            computing_data_2.diff_count = diff_count;
            computing_data_2.gain_type = gain_type;
            computing_data_2.sep_gain = sep_gain;
            computing_data_2.normalization = normalization;
            computing_data_2.biased_IC = biased_IC;
            computing_data_2.diff_vector = diff_vector;
            computing_data_2.neuron_vector = num_neuron_vector;
            computing_data_2.IC_vector = IC_vector_2;

            threads_2.push_back(thread(workLoop_2,computing_data_2,ref(W_Diff_N_Vector),ref(W_Diff_W_Vector),
            ref(W_Diff_diff_Vector),ref(W_Diff_Acc_Vector),ref(W_Diff_AccMean_Vector),ref(W_Diff_RT_Vector), ref(W_Diff_Acc_Var_Vector),
        ref(W_Diff_RT_Var_Vector)));
        }

        for (int i=0;i<threads_2.size();i++){
            threads_2[i].join();
        }

        //Writing W vs. Diff data
        // vector<string> gain_names_2;
        // vector<string> sep_names_2;
        // vector<string> norm_names_2;
        // vector<string> biased_names_2;
        vector<string> N_names;
        N_names.push_back("_N=10");
        N_names.push_back("_N=50");
        N_names.push_back("_N=100");
        // gain_names_2.push_back("_Sigm");
        // gain_names_2.push_back("_Binary");
        // sep_names_2.push_back("_IntAll");
        // sep_names_2.push_back("_Sep");
        // norm_names_2.push_back("_YesNorm");
        // norm_names_2.push_back("_NoNorm");
        // biased_names_2.push_back("_Fair");
        // biased_names_2.push_back("_Unfair");

        index=0;

        for (int d=0;d<num_neuron_count;d++){
            for (int g=0;g<gain_type;g++){
                for (int s=0;s<sep_gain;s++){
                    for (int n=0;n<normalization;n++){
                        for (int b=0;b<biased_IC;b++){
                            for (int dist=0;dist<2;dist++){
                                string filename1 = "Data/" + base_name +"_N"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b]+ dist_names[dist] + N_names[d] +  ".txt";
                                string filename2 = "Data/" +base_name +"_W"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b]+dist_names[dist] + N_names[d] +   ".txt";
                                string filename3 = "Data/" +base_name +"_Diff"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b]+dist_names[dist] + N_names[d] +   ".txt";
                                string filename4 = "Data/" +base_name +"_Acc"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b]+dist_names[dist] + N_names[d] +   ".txt";
                                string filename5 = "Data/" +base_name +"_RT"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b] + dist_names[dist] + N_names[d] +  ".txt";
                                string filename6 = "Data/" +base_name +"_Acc_Var"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b] + dist_names[dist] + N_names[d] + ".txt";
                                string filename7 = "Data/" +base_name +"_RT_Var"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b] + dist_names[dist] + N_names[d] +  ".txt";
                                string filename8 = "Data/" +base_name +"_AccMean"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b] + dist_names[dist] + N_names[d] +  ".txt";
                                ofstream ToWrite1(filename1);
                                ofstream ToWrite2(filename2);
                                ofstream ToWrite3(filename3);
                                ofstream ToWrite4(filename4);
                                ofstream ToWrite5(filename5);
                                ofstream ToWrite6(filename6);
                                ofstream ToWrite7(filename7);
                                ofstream ToWrite8(filename8);
                                //int index = b*1 + n*2 + s*4 + g*8;
                                for (int i=0;i<W_Diff_N_Vector[index].size();i++){
                                    ToWrite1 << W_Diff_N_Vector[index][i] << endl;
                                    ToWrite2 << W_Diff_W_Vector[index][i] << endl;
                                    ToWrite3 << W_Diff_diff_Vector[index][i] << endl;
                                    ToWrite4 << W_Diff_Acc_Vector[index][i] << endl;
                                    ToWrite5 << W_Diff_RT_Vector[index][i] << endl;
                                    ToWrite6 << W_Diff_Acc_Var_Vector[index][i] << endl;
                                    ToWrite7 << W_Diff_RT_Var_Vector[index][i] << endl;
                                    ToWrite8 << W_Diff_AccMean_Vector[index][i] << endl;
                                }
                                index++;
                            }
                        }
                    }
                }
            }
        }
    }






    //Behold one last time, the ultimate version for W vs. P
    if (run_W_P){
        int num_outer_loop_ult_3 = 11;
        int num_inner_loop_ult_3 = 14;
        int update_times_ult_3 = 10000;

        int num_Fair_IC_ult_3 = 10;
        int num_Unfair_IC_ult_3 = 100;
        vector<int> IC_vector_3;
        IC_vector_3.push_back(num_Fair_IC_ult_3);
        IC_vector_3.push_back(num_Unfair_IC_ult_3);

        vector<vector<double> > W_P_N_Vector;
        vector<vector<double> > W_P_P_Vector;
        vector<vector<double> > W_P_W_Vector;
        vector<vector<double> > W_P_diff_Vector;
        vector<vector<double> > W_P_Acc_Vector;
        vector<vector<double> > W_P_AccMean_Vector;
        vector<vector<double> > W_P_RT_Vector;
        vector<vector<double> > W_P_Acc_Var_Vector;
        vector<vector<double> > W_P_RT_Var_Vector;

        for (int i=0;i<num_neuron_count*gain_type*sep_gain*normalization*biased_IC*distributed_input;i++){
            vector<double> toPush;
            W_P_N_Vector.push_back(toPush);
            W_P_P_Vector.push_back(toPush);
            W_P_W_Vector.push_back(toPush);
            W_P_diff_Vector.push_back(toPush);
            W_P_Acc_Vector.push_back(toPush);
            W_P_AccMean_Vector.push_back(toPush);
            W_P_RT_Vector.push_back(toPush);
            W_P_Acc_Var_Vector.push_back(toPush);
            W_P_RT_Var_Vector.push_back(toPush);
        }

        index = 0;
        vector<thread> threads_3;

        Data computing_data_3;

        for (int i=1;i<=num_outer_loop_ult_3;i++){
            double p = pow(2,i-num_outer_loop_ult_3);
            computing_data_3.N = 300;
            computing_data_3.diff = 0;
            computing_data_3.p = p;
            computing_data_3.k = 0;
            computing_data_3.update_times = update_times_ult_3;
            computing_data_3.num_inner_loop_ult = num_inner_loop_ult_3;
            computing_data_3.diff_count = diff_count;
            computing_data_3.gain_type = gain_type;
            computing_data_3.sep_gain = sep_gain;
            computing_data_3.normalization = normalization;
            computing_data_3.biased_IC = biased_IC;
            computing_data_3.diff_vector = diff_vector;
            computing_data_3.neuron_vector = num_neuron_vector;
            computing_data_3.IC_vector = IC_vector_3;

            threads_3.push_back(thread(workLoop_3,computing_data_3,ref(W_P_N_Vector),ref(W_P_P_Vector), ref(W_P_W_Vector),
            ref(W_P_diff_Vector),ref(W_P_Acc_Vector),ref(W_P_AccMean_Vector),ref(W_P_RT_Vector), ref(W_P_Acc_Var_Vector),
        ref(W_P_RT_Var_Vector)));
        }

        for (int i=0;i<threads_3.size();i++){
            threads_3[i].join();
        }


        //Data outputting
        string base_name_2 = "W_P";

        index=0;

        for (int d=0;d<num_neuron_count;d++){
            for (int g=0;g<gain_type;g++){
                for (int s=0;s<sep_gain;s++){
                    for (int n=0;n<normalization;n++){
                        for (int b=0;b<biased_IC;b++){
                            for (int dist=0;dist<2;dist++){
                                string filename1 = "Data/" + base_name_2 +"_N"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b]+ dist_names[dist] + diff_names[d] + ".txt";
                                string filename2 = "Data/" +base_name_2 +"_W"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b]+dist_names[dist] + diff_names[d] +  ".txt";
                                string filename3 = "Data/" +base_name_2 +"_Diff"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b]+dist_names[dist] + diff_names[d] +  ".txt";
                                string filename4 = "Data/" +base_name_2 +"_Acc"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b]+dist_names[dist] + diff_names[d] +  ".txt";
                                string filename5 = "Data/" +base_name_2 +"_RT"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b] + dist_names[dist] + diff_names[d] + ".txt";
                                string filename6 = "Data/" +base_name_2 +"_Acc_Var"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b] + dist_names[dist] + diff_names[d] + ".txt";
                                string filename7 = "Data/" +base_name_2 +"_RT_Var"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b] + dist_names[dist] + diff_names[d] + ".txt";
                                string filename8 = "Data/" +base_name_2 +"_P"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b] + dist_names[dist] + diff_names[d] + ".txt";
                                string filename9 = "Data/" +base_name_2 +"_AccMean"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b] + dist_names[dist] + diff_names[d] + ".txt";
                                ofstream ToWrite1(filename1);
                                ofstream ToWrite2(filename2);
                                ofstream ToWrite3(filename3);
                                ofstream ToWrite4(filename4);
                                ofstream ToWrite5(filename5);
                                ofstream ToWrite6(filename6);
                                ofstream ToWrite7(filename7);
                                ofstream ToWrite8(filename8);
                                ofstream ToWrite9(filename9);
                                //int index = b*1 + n*2 + s*4 + g*8;
                                for (int i=0;i<W_P_N_Vector[index].size();i++){
                                    ToWrite1 << W_P_N_Vector[index][i] << endl;
                                    ToWrite2 << W_P_W_Vector[index][i] << endl;
                                    ToWrite3 << W_P_diff_Vector[index][i] << endl;
                                    ToWrite4 << W_P_Acc_Vector[index][i] << endl;
                                    ToWrite5 << W_P_RT_Vector[index][i] << endl;
                                    ToWrite6 << W_P_Acc_Var_Vector[index][i] << endl;
                                    ToWrite7 << W_P_RT_Var_Vector[index][i] << endl;
                                    ToWrite8 << W_P_P_Vector[index][i] << endl;
                                    ToWrite9 << W_P_AccMean_Vector[index][i] << endl;
                                }
                                index++;
                            }
                        }
                    }
                }
            }
        }
    }


    //From Regular to All-to-All
    if (run_W_Regular){
        int num_outer_loop_ult_4 = 15;
        int num_inner_loop_ult_4 = 14;
        int update_times_ult_4 = 1000;

        int num_Fair_IC_ult_4 = 10;
        int num_Unfair_IC_ult_4 = 100;
        vector<int> IC_vector_4;
        IC_vector_4.push_back(num_Fair_IC_ult_4);
        IC_vector_4.push_back(num_Unfair_IC_ult_4);

        vector<vector<double> > W_Regular_N_Vector;
        vector<vector<double> > W_Regular_P_Vector;
        vector<vector<double> > W_Regular_K_Vector;
        vector<vector<double> > W_Regular_W_Vector;
        vector<vector<double> > W_Regular_diff_Vector;
        vector<vector<double> > W_Regular_Acc_Vector;
        vector<vector<double> > W_Regular_AccMean_Vector;
        vector<vector<double> > W_Regular_RT_Vector;
        vector<vector<double> > W_Regular_Acc_Var_Vector;
        vector<vector<double> > W_Regular_RT_Var_Vector;

        for (int i=0;i<num_neuron_count*gain_type*sep_gain*normalization*biased_IC*distributed_input;i++){
            vector<double> toPush;
            W_Regular_N_Vector.push_back(toPush);
            W_Regular_P_Vector.push_back(toPush);
            W_Regular_K_Vector.push_back(toPush);
            W_Regular_W_Vector.push_back(toPush);
            W_Regular_diff_Vector.push_back(toPush);
            W_Regular_Acc_Vector.push_back(toPush);
            W_Regular_AccMean_Vector.push_back(toPush);
            W_Regular_RT_Vector.push_back(toPush);
            W_Regular_Acc_Var_Vector.push_back(toPush);
            W_Regular_RT_Var_Vector.push_back(toPush);
        }

        index = 0;
        vector<thread> threads_4;

        Data computing_data_4;

        for (int i=1;i<=num_outer_loop_ult_4;i++){
            int k=0;
            if (i <=5){
              k = i*5;
            }
            else{
              k = (i-5)*30;
            }
            computing_data_4.N = 300;
            computing_data_4.diff = 0;
            computing_data_4.p = 0;
            computing_data_4.k = k;
            computing_data_4.update_times = update_times_ult_4;
            computing_data_4.num_inner_loop_ult = num_inner_loop_ult_4;
            computing_data_4.diff_count = diff_count;
            computing_data_4.gain_type = gain_type;
            computing_data_4.sep_gain = sep_gain;
            computing_data_4.normalization = normalization;
            computing_data_4.biased_IC = biased_IC;
            computing_data_4.diff_vector = diff_vector;
            computing_data_4.neuron_vector = num_neuron_vector;
            computing_data_4.IC_vector = IC_vector_4;

            threads_4.push_back(thread(workLoop_4,computing_data_4,ref(W_Regular_N_Vector),ref(W_Regular_P_Vector),
            ref(W_Regular_K_Vector),ref(W_Regular_W_Vector),
            ref(W_Regular_diff_Vector),ref(W_Regular_Acc_Vector),ref(W_Regular_AccMean_Vector),ref(W_Regular_RT_Vector), ref(W_Regular_Acc_Var_Vector),
        ref(W_Regular_RT_Var_Vector)));
        }

        for (int i=0;i<threads_4.size();i++){
            threads_4[i].join();
        }

        //Data outputting
        string base_name_3 = "W_Regular";

        index=0;

        for (int d=0;d<num_neuron_count;d++){
            for (int g=0;g<gain_type;g++){
                for (int s=0;s<sep_gain;s++){
                    for (int n=0;n<normalization;n++){
                        for (int b=0;b<biased_IC;b++){
                            for (int dist=0;dist<2;dist++){
                                string filename1 = "Data/" + base_name_3 +"_N"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b]+ dist_names[dist] + diff_names[d] + ".txt";
                                string filename2 = "Data/" +base_name_3 +"_W"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b]+dist_names[dist] + diff_names[d] +  ".txt";
                                string filename3 = "Data/" +base_name_3 +"_Diff"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b]+dist_names[dist] + diff_names[d] +  ".txt";
                                string filename4 = "Data/" +base_name_3 +"_Acc"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b]+dist_names[dist] + diff_names[d] +  ".txt";
                                string filename5 = "Data/" +base_name_3 +"_RT"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b] + dist_names[dist] + diff_names[d] + ".txt";
                                string filename6 = "Data/" +base_name_3 +"_Acc_Var"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b] + dist_names[dist] + diff_names[d] + ".txt";
                                string filename7 = "Data/" +base_name_3 +"_RT_Var"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b] + dist_names[dist] + diff_names[d] + ".txt";
                                string filename8 = "Data/" +base_name_3 +"_P"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b] + dist_names[dist] + diff_names[d] + ".txt";
                                string filename9 = "Data/" +base_name_3 +"_K"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b] + dist_names[dist] + diff_names[d] + ".txt";
                                string filename10 = "Data/" +base_name_3 +"_AccMean"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b] + dist_names[dist] + diff_names[d] + ".txt";
                                ofstream ToWrite1(filename1);
                                ofstream ToWrite2(filename2);
                                ofstream ToWrite3(filename3);
                                ofstream ToWrite4(filename4);
                                ofstream ToWrite5(filename5);
                                ofstream ToWrite6(filename6);
                                ofstream ToWrite7(filename7);
                                ofstream ToWrite8(filename8);
                                ofstream ToWrite9(filename9);
                                ofstream ToWrite10(filename10);
                                //int index = b*1 + n*2 + s*4 + g*8;
                                for (int i=0;i<W_Regular_N_Vector[index].size();i++){
                                    ToWrite1 << W_Regular_N_Vector[index][i] << endl;
                                    ToWrite2 << W_Regular_W_Vector[index][i] << endl;
                                    ToWrite3 << W_Regular_diff_Vector[index][i] << endl;
                                    ToWrite4 << W_Regular_Acc_Vector[index][i] << endl;
                                    ToWrite5 << W_Regular_RT_Vector[index][i] << endl;
                                    ToWrite6 << W_Regular_Acc_Var_Vector[index][i] << endl;
                                    ToWrite7 << W_Regular_RT_Var_Vector[index][i] << endl;
                                    ToWrite8 << W_Regular_P_Vector[index][i] << endl;
                                    ToWrite9 << W_Regular_K_Vector[index][i] << endl;
                                    ToWrite10 << W_Regular_AccMean_Vector[index][i] << endl;
                                }
                                index++;
                            }
                        }
                    }
                }
            }
        }
    }


    //From Random to All-to-All
    if (run_W_Random){
        int num_outer_loop_ult_5 = 15;
        int num_inner_loop_ult_5 = 14;
        int update_times_ult_5 = 500;

        int num_Fair_IC_ult_5 = 10;
        int num_Unfair_IC_ult_5 = 100;
        vector<int> IC_vector_5;
        IC_vector_5.push_back(num_Fair_IC_ult_5);
        IC_vector_5.push_back(num_Unfair_IC_ult_5);

        vector<vector<double> > W_Random_N_Vector;
        vector<vector<double> > W_Random_P_Vector;
        vector<vector<double> > W_Random_K_Vector;
        vector<vector<double> > W_Random_W_Vector;
        vector<vector<double> > W_Random_diff_Vector;
        vector<vector<double> > W_Random_Acc_Vector;
        vector<vector<double> > W_Random_AccMean_Vector;
        vector<vector<double> > W_Random_RT_Vector;
        vector<vector<double> > W_Random_Acc_Var_Vector;
        vector<vector<double> > W_Random_RT_Var_Vector;

        for (int i=0;i<num_neuron_count*gain_type*sep_gain*normalization*biased_IC*distributed_input;i++){
            vector<double> toPush;
            W_Random_N_Vector.push_back(toPush);
            W_Random_P_Vector.push_back(toPush);
            W_Random_K_Vector.push_back(toPush);
            W_Random_W_Vector.push_back(toPush);
            W_Random_diff_Vector.push_back(toPush);
            W_Random_Acc_Vector.push_back(toPush);
            W_Random_AccMean_Vector.push_back(toPush);
            W_Random_RT_Vector.push_back(toPush);
            W_Random_Acc_Var_Vector.push_back(toPush);
            W_Random_RT_Var_Vector.push_back(toPush);
        }

        index = 0;
        vector<thread> threads_5;

        Data computing_data_5;

        for (int i=1;i<=num_outer_loop_ult_5;i++){
            double p =0;
            if (i<=5){
                p = i*5/300;
            }
            else{
                p = (i-5)*0.1;
            }


            computing_data_5.N = 300;
            computing_data_5.diff = 0;
            computing_data_5.p = p;
            computing_data_5.k = 0;
            computing_data_5.update_times = update_times_ult_5;
            computing_data_5.num_inner_loop_ult = num_inner_loop_ult_5;
            computing_data_5.diff_count = diff_count;
            computing_data_5.gain_type = gain_type;
            computing_data_5.sep_gain = sep_gain;
            computing_data_5.normalization = normalization;
            computing_data_5.biased_IC = biased_IC;
            computing_data_5.diff_vector = diff_vector;
            computing_data_5.neuron_vector = num_neuron_vector;
            computing_data_5.IC_vector = IC_vector_5;

            threads_5.push_back(thread(workLoop_5,computing_data_5,ref(W_Random_N_Vector),ref(W_Random_P_Vector),
            ref(W_Random_K_Vector), ref(W_Random_W_Vector),
            ref(W_Random_diff_Vector),ref(W_Random_Acc_Vector),ref(W_Random_AccMean_Vector),ref(W_Random_RT_Vector), ref(W_Random_Acc_Var_Vector),
        ref(W_Random_RT_Var_Vector)));
        }

        for (int i=0;i<threads_5.size();i++){
            threads_5[i].join();
        }

        //Data outputting
        base_name = "W_Random";

        index=0;

        for (int d=0;d<num_neuron_count;d++){
            for (int g=0;g<gain_type;g++){
                for (int s=0;s<sep_gain;s++){
                    for (int n=0;n<normalization;n++){
                        for (int b=0;b<biased_IC;b++){
                            for (int dist=0;dist<2;dist++){
                                string filename1 = "Data/" + base_name +"_N"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b]+ dist_names[dist] + diff_names[d] + ".txt";
                                string filename2 = "Data/" +base_name +"_W"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b]+dist_names[dist] + diff_names[d] +  ".txt";
                                string filename3 = "Data/" +base_name +"_Diff"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b]+dist_names[dist] + diff_names[d] +  ".txt";
                                string filename4 = "Data/" +base_name +"_Acc"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b]+dist_names[dist] + diff_names[d] +  ".txt";
                                string filename5 = "Data/" +base_name +"_RT"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b] + dist_names[dist] + diff_names[d] + ".txt";
                                string filename6 = "Data/" +base_name +"_Acc_Var"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b] + dist_names[dist] + diff_names[d] + ".txt";
                                string filename7 = "Data/" +base_name +"_RT_Var"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b] + dist_names[dist] + diff_names[d] + ".txt";
                                string filename8 = "Data/" +base_name +"_P"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b] + dist_names[dist] + diff_names[d] + ".txt";
                                string filename9 = "Data/" +base_name +"_K"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b] + dist_names[dist] + diff_names[d] + ".txt";
                                string filename10 = "Data/" +base_name +"_AccMean"+ gain_names[g] + sep_names[s]
                                + norm_names[n] + biased_names[b] + dist_names[dist] + diff_names[d] + ".txt";
                                ofstream ToWrite1(filename1);
                                ofstream ToWrite2(filename2);
                                ofstream ToWrite3(filename3);
                                ofstream ToWrite4(filename4);
                                ofstream ToWrite5(filename5);
                                ofstream ToWrite6(filename6);
                                ofstream ToWrite7(filename7);
                                ofstream ToWrite8(filename8);
                                ofstream ToWrite9(filename9);
                                ofstream ToWrite10(filename10);
                                //int index = b*1 + n*2 + s*4 + g*8;
                                for (int i=0;i<W_Random_N_Vector[index].size();i++){
                                    ToWrite1 << W_Random_N_Vector[index][i] << endl;
                                    ToWrite2 << W_Random_W_Vector[index][i] << endl;
                                    ToWrite3 << W_Random_diff_Vector[index][i] << endl;
                                    ToWrite4 << W_Random_Acc_Vector[index][i] << endl;
                                    ToWrite5 << W_Random_RT_Vector[index][i] << endl;
                                    ToWrite6 << W_Random_Acc_Var_Vector[index][i] << endl;
                                    ToWrite7 << W_Random_RT_Var_Vector[index][i] << endl;
                                    ToWrite8 << W_Random_P_Vector[index][i] << endl;
                                    ToWrite9 << W_Random_K_Vector[index][i] << endl;
                                    ToWrite10 << W_Random_AccMean_Vector[index][i] << endl;
                                }
                                index++;
                            }
                        }
                    }
                }
            }
        }
    }



    //Outputting
    ofstream timeData("Data/Time.txt");
    ofstream x1Data("Data/x1.txt");
    ofstream x2Data("Data/x2.txt");
    ofstream x3Data("Data/x3.txt");
    ofstream x4Data("Data/x4.txt");
    ofstream x5Data("Data/x5.txt");
    ofstream xWinnerData("Data/x_w.txt");
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

    // ofstream W_N_W_Data_ult("Data/W_N_W_ult.txt");
    // ofstream W_N_N_Data_ult("Data/W_N_N_ult.txt");
    // ofstream W_N_diff_Data_ult("Data/W_N_diff_ult.txt");
    // ofstream W_N_Acc_Data_ult("Data/W_N_Acc_ult.txt");
    // ofstream W_N_RT_Data_ult("Data/W_N_RT_ult.txt");

    for (int i=0;i<n1->t_data.size();i++){
        timeData << n1->t_data[i] << endl;
        x1Data << n1->x_data[i] << endl;
        x2Data << n2->x_data[i] << endl;
        x3Data << n3->x_data[i] << endl;
        x4Data << n4->x_data[i] << endl;
        x5Data << n5->x_data[i] << endl;
        xWinnerData << n_winner->x_data[i] << endl;
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



    // int index = 0;
    // for (int d=0;d<diff_count;d++){
    //     double diff_ult = diff_vector[d];
    //     for (int g=0;g<gain_type;g++){
    //         for (int s=0;s<sep_gain;s++){
    //             for (int n=0;n<normalization;n++){
    //                 for (int b=0;b<biased_IC;b++){
    //                     int num_IC = IC_vector[b];
    //                     for (int i=1;i<=num_outer_loop_ult;i++){
    //                         //double N = 10;
    //                         double N = i*10;
    //                         for (int j=1;j<=num_inner_loop_ult;j++){
    //                             //double w = 1.0;
    //                             double Acc_sum =0; //IC avg.
    //                             double RT_sum = 0; //IC avg.
    //                             double W = (double)(j-1);
    //                             for (int l=1;l<=num_IC;l++){
    //                                 double IC = 0.1 + 0.8/num_IC * l;
    //                                 //cout << "here" <<endl;
    //                                 Network network_3D(N,W,1,0.1,n);
    //                                 network_3D.constructAllToAllNetwork();
    //                                 network_3D.initializeWithChoice(b,IC,diff_ult);
    //                                 //network_3D.initialize(0.5);
    //                                 for (int k = 0;k<update_times_ult;k++){
    //                                     network_3D.updateWithChoice(s,g);
    //                                     if (k==update_times_ult-1){
    //                                         network_3D.computeAccuracy();
    //                                     }
    //                                 }
    //                                 Acc_sum += max(0,network_3D.Acc); //if Acc is neg, set to 0.
    //                                 RT_sum += network_3D.RT;
    //                             }
    //                             //data collection
    //                             //int index = b*1 + n*2 + s*4 + g*8;
    //                             W_N_N_Vector[index].push_back(N);
    //                             W_N_W_Vector[index].push_back(W);
    //                             W_N_diff_Vector[index].push_back(diff_ult);
    //                             W_N_Acc_Vector[index].push_back(Acc_sum/num_IC);
    //                             W_N_RT_Vector[index].push_back(RT_sum/num_IC);
    //                         }
    //                     }
    //                     //update index out of the simulation loop
    //                     index++;
    //                 }
    //             }
    //         }
    //     }
    // }

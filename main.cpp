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
    int update_times_3D = 10;

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
    int num_outer_loop_AVG_Fair = 20;
    int num_inner_loop_AVG_Fair = 11;
    int update_times_3D_AVG_Fair = 5;
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
        double diff = 0.8;
        for (int j=1;j<=num_inner_loop_AVG_Fair;j++){
            double Acc_sum =0; //IC avg.
            double RT_sum = 0; //IC avg.
            double w = (double)(j-1);
            for (int l=1;l<=num_IC_AVG_Fair;l++){
                double IC = 0.1 + 0.8/num_IC_AVG_Fair * l;
                //double IC = l*0.09;
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

    int num_outer_loop_ult = 20;
    int num_inner_loop_ult = 11;
    int update_times_ult = 5;
    //double diff_ult = 0.5;
    vector<double> diff_vector;
    diff_vector.push_back(0.2);
    diff_vector.push_back(0.5);
    diff_vector.push_back(0.8);

    int num_Fair_IC_ult = 10;
    int num_Unfair_IC_ult = 1;
    vector<int> IC_vector;
    IC_vector.push_back(num_Fair_IC_ult);
    IC_vector.push_back(num_Unfair_IC_ult);

    vector<vector<double> > W_N_N_Vector;
    vector<vector<double> > W_N_W_Vector;
    vector<vector<double> > W_N_diff_Vector;
    vector<vector<double> > W_N_Acc_Vector;
    vector<vector<double> > W_N_RT_Vector;
    
    for (int i=0;i<diff_count*gain_type*sep_gain*normalization*biased_IC;i++){
        vector<double> toPush;
        W_N_N_Vector.push_back(toPush);
        W_N_W_Vector.push_back(toPush);
        W_N_diff_Vector.push_back(toPush);
        W_N_Acc_Vector.push_back(toPush);
        W_N_RT_Vector.push_back(toPush);
    }

    int index = 0;
    for (int d=0;d<diff_count;d++){
        double diff_ult = diff_vector[d];
        for (int g=0;g<gain_type;g++){
            for (int s=0;s<sep_gain;s++){
                for (int n=0;n<normalization;n++){
                    for (int b=0;b<biased_IC;b++){
                        int num_IC = IC_vector[b];
                        for (int i=1;i<=num_outer_loop_ult;i++){
                            //double N = 10;
                            double N = i*10;
                            for (int j=1;j<=num_inner_loop_ult;j++){
                                //double w = 1.0;
                                double Acc_sum =0; //IC avg.
                                double RT_sum = 0; //IC avg.
                                double W = (double)(j-1);
                                for (int l=1;l<=num_IC;l++){
                                    double IC = 0.1 + 0.8/num_IC * l;
                                    //cout << "here" <<endl;
                                    Network network_3D(N,W,1,0.01,b);
                                    network_3D.constructAllToAllNetwork();
                                    network_3D.initializeWithChoice(b,IC,diff_ult);
                                    //network_3D.initialize(0.5);
                                    for (int k = 0;k<update_times_ult;k++){
                                        network_3D.updateWithChoice(s,g);
                                        if (k==update_times_ult-1){
                                            network_3D.computeAccuracy();
                                        }
                                    }
                                    Acc_sum += max(0,network_3D.Acc); //if Acc is neg, set to 0.
                                    RT_sum += network_3D.RT;                               
                                }
                                //data collection
                                //int index = b*1 + n*2 + s*4 + g*8;
                                W_N_N_Vector[index].push_back(N);
                                W_N_W_Vector[index].push_back(W);
                                W_N_diff_Vector[index].push_back(diff_ult);
                                W_N_Acc_Vector[index].push_back(Acc_sum/num_IC);
                                W_N_RT_Vector[index].push_back(RT_sum/num_IC);
                            }
                        }
                        //update index out of the simulation loop
                        index++;
                    }
                }
            }
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

    // ofstream W_N_W_Data_ult("Data/W_N_W_ult.txt");
    // ofstream W_N_N_Data_ult("Data/W_N_N_ult.txt");
    // ofstream W_N_diff_Data_ult("Data/W_N_diff_ult.txt");
    // ofstream W_N_Acc_Data_ult("Data/W_N_Acc_ult.txt");
    // ofstream W_N_RT_Data_ult("Data/W_N_RT_ult.txt");

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

    // for (int i=0;i<W_N_W_Vector.size();i++){
    //     for (int j=0;j<W_N_W_Vector[i].size();j++){
    //         W_N_W_Data_ult << W_N_W_Vector[i][j] << endl;
    //         W_N_N_Data_ult << W_N_N_Vector[i][j] << endl;
    //         W_N_diff_Data_ult << W_N_diff_Vector[i][j] << endl;
    //         W_N_Acc_Data_ult << W_N_Acc_Vector[i][j] << endl;
    //         W_N_RT_Data_ult << W_N_RT_Vector[i][j] << endl;
    //     }
    // }

    vector<string> diff_names;
    vector<string> gain_names;
    vector<string> sep_names;
    vector<string> norm_names;
    vector<string> biased_names;
    diff_names.push_back("_Hard");
    diff_names.push_back("_Medium");
    diff_names.push_back("_Easy");
    gain_names.push_back("_Sigm");
    gain_names.push_back("_Binary");
    sep_names.push_back("_IntAll");
    sep_names.push_back("_Sep");
    norm_names.push_back("_YesNorm");
    norm_names.push_back("_NoNorm");
    biased_names.push_back("_Fair");
    biased_names.push_back("_Unfair");
    string W_N_base_name = "W_N";

    index=0;
    
    for (int d=0;d<diff_count;d++){
        for (int g=0;g<gain_type;g++){
            for (int s=0;s<sep_gain;s++){
                for (int n=0;n<normalization;n++){
                    for (int b=0;b<biased_IC;b++){
                        string filename1 = "Data/" + W_N_base_name +"_N"+ gain_names[g] + sep_names[s]
                        + norm_names[n] + biased_names[b]+ diff_names[d] + ".txt";
                        string filename2 = "Data/" +W_N_base_name +"_W"+ gain_names[g] + sep_names[s]
                        + norm_names[n] + biased_names[b]+diff_names[d] +  ".txt";
                        string filename3 = "Data/" +W_N_base_name +"_diff"+ gain_names[g] + sep_names[s]
                        + norm_names[n] + biased_names[b]+diff_names[d] +  ".txt";
                        string filename4 = "Data/" +W_N_base_name +"_Acc"+ gain_names[g] + sep_names[s]
                        + norm_names[n] + biased_names[b]+diff_names[d] +  ".txt";
                        string filename5 = "Data/" +W_N_base_name +"_RT"+ gain_names[g] + sep_names[s]
                        + norm_names[n] + biased_names[b] + diff_names[d] + ".txt";
                        ofstream ToWrite1(filename1);
                        ofstream ToWrite2(filename2);
                        ofstream ToWrite3(filename3);
                        ofstream ToWrite4(filename4);
                        ofstream ToWrite5(filename5);
                        //int index = b*1 + n*2 + s*4 + g*8;
                        for (int i=0;i<W_N_N_Vector[index].size();i++){
                            ToWrite1 << W_N_N_Vector[index][i] << endl;
                            ToWrite2 << W_N_W_Vector[index][i] << endl;
                            ToWrite3 << W_N_diff_Vector[index][i] << endl;
                            ToWrite4 << W_N_Acc_Vector[index][i] << endl;
                            ToWrite5 << W_N_RT_Vector[index][i] << endl;
                        }
                        index++;
                    }
                }
            }
        }
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

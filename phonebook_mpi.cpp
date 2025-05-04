/*
    How to run:
    mpic++ -o search phonebook_search.cpp
    mpirun -np 4 ./search phonebook1.txt Bob
*/

//Simplified
#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;

struct Contact {
    string name;
    string phone;
};

void send_string(const string &text, int receiver) {
    int len = text.size() + 1;
    MPI_Send(&len, 1, MPI_INT, receiver, 1, MPI_COMM_WORLD);
    MPI_Send(text.c_str(), len, MPI_CHAR, receiver, 1, MPI_COMM_WORLD);
}

string receive_string(int sender) {
    int len;
    MPI_Recv(&len, 1, MPI_INT, sender, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    char *buf = new char[len];
    MPI_Recv(buf, len, MPI_CHAR, sender, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    string res(buf);
    delete[] buf;
    return res;
}

string vector_to_string(const vector<Contact> &contacts, int start, int end) {
    string result;
    for (int i = start; i < min((int)contacts.size(), end); i++) {
        result += contacts[i].name + "," + contacts[i].phone + "\n";
    }
    return result;
}

vector<Contact> string_to_contacts(const string &text) {
    vector<Contact> contacts;
    istringstream iss(text);
    string line;
    while (getline(iss, line)) {
        if (line.empty()) continue;
        int comma = line.find(",");
        if (comma == string::npos) continue;
        contacts.push_back({line.substr(0, comma), line.substr(comma + 1)});
    }
    return contacts;
}

string check(const Contact &c, const string &search) {
    if (c.name.find(search) != string::npos) {
        return c.name + " " + c.phone + "\n";
    }
    return "";
}

void read_phonebook(const vector<string> &files, vector<Contact> &contacts) {
    for (const string &file : files) {
        ifstream f(file);
        string line;
        while (getline(f, line)) {
            if (line.empty()) continue;
            int comma = line.find(",");
            if (comma == string::npos) continue;
            contacts.push_back({line.substr(1, comma - 2), line.substr(comma + 2, line.size() - comma - 3)});
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0)
            cerr << "Usage: mpirun -n <procs> " << argv[0] << " <file>... <search>\n";
        MPI_Finalize();
        return 1;
    }

    string search_term = argv[argc - 1];
    double start, end;

    if (rank == 0) {
        vector<string> files(argv + 1, argv + argc - 1);
        vector<Contact> contacts;
        read_phonebook(files, contacts);
        int total = contacts.size();
        int chunk = (total + size - 1) / size;

        for (int i = 1; i < size; i++) {
            string text = vector_to_string(contacts, i * chunk, (i + 1) * chunk);
            send_string(text, i);
        }

        start = MPI_Wtime();
        string result;
        for (int i = 0; i < min(chunk, total); i++) {
            string match = check(contacts[i], search_term);
            if (!match.empty()) result += match;
        }
        end = MPI_Wtime();

        for (int i = 1; i < size; i++) {
            string recv = receive_string(i);
            if (!recv.empty()) result += recv;
        }
        
        ofstream out("output.txt");
        out << result;
        out.close();
        printf("Process %d took %f seconds.\n", rank, end - start);

    } else {
        string recv_text = receive_string(0);
        vector<Contact> contacts = string_to_contacts(recv_text);
        start = MPI_Wtime();
        string result;
        for (auto &c : contacts) {
            string match = check(c, search_term);
            if (!match.empty()) result += match;
        }
        end = MPI_Wtime();
        send_string(result, 0);
        printf("Process %d took %f seconds.\n", rank, end - start);
    }

    MPI_Finalize();
    return 0;
}

/*
    How to compile and run:
    Compile: mpic++ -o search phonebook_search.cpp
    Run:     mpirun -np 4 ./search phonebook1.txt Bob

    This program performs a parallel search for a name (e.g., "Bob")
    in a phonebook file using MPI. It divides the workload among processes
    to speed up the search operation.
*/

#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;

// Struct to represent a contact entry with name and phone number
struct Contact {
    string name;
    string phone;
};

// Function to send a string to a specific receiver process
void send_string(const string &text, int receiver) {
    int len = text.size() + 1; // +1 for null terminator
    MPI_Send(&len, 1, MPI_INT, receiver, 1, MPI_COMM_WORLD);           // Send length first
    MPI_Send(text.c_str(), len, MPI_CHAR, receiver, 1, MPI_COMM_WORLD); // Then send the actual string
}

// Function to receive a string from a specific sender process
string receive_string(int sender) {
    int len;
    MPI_Recv(&len, 1, MPI_INT, sender, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Receive length
    char *buf = new char[len];
    MPI_Recv(buf, len, MPI_CHAR, sender, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Receive string
    string res(buf);
    delete[] buf;
    return res;
}

// Converts a vector of contacts to a string format for sending via MPI
string vector_to_string(const vector<Contact> &contacts, int start, int end) {
    string result;
    for (int i = start; i < min((int)contacts.size(), end); i++) {
        result += contacts[i].name + "," + contacts[i].phone + "\n";
    }
    return result;
}

// Converts a received string back into a vector of Contact objects
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

// Check if the contact name contains the search term
string check(const Contact &c, const string &search) {
    if (c.name.find(search) != string::npos) {
        return c.name + " " + c.phone + "\n";
    }
    return "";
}

// Reads contacts from one or more phonebook files into a vector
void read_phonebook(const vector<string> &files, vector<Contact> &contacts) {
    for (const string &file : files) {
        ifstream f(file);
        string line;
        while (getline(f, line)) {
            if (line.empty()) continue;
            int comma = line.find(",");
            if (comma == string::npos) continue;
            // Clean and extract name and phone
            contacts.push_back({line.substr(1, comma - 2), line.substr(comma + 2, line.size() - comma - 3)});
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);                         // Initialize the MPI environment
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);           // Get current process ID
    MPI_Comm_size(MPI_COMM_WORLD, &size);           // Get total number of processes

    // Check if the user provided sufficient arguments
    if (argc < 3) {
        if (rank == 0)
            cerr << "Usage: mpirun -n <procs> " << argv[0] << " <file>... <search_term>\n";
        MPI_Finalize();
        return 1;
    }

    string search_term = argv[argc - 1]; // Last argument is the search term
    double start, end;

    // Master process (rank 0) handles reading and distributing the workload
    if (rank == 0) {
        // Gather all input file names (excluding search term)
        vector<string> files(argv + 1, argv + argc - 1);
        vector<Contact> contacts;

        read_phonebook(files, contacts);            // Read contacts from the files
        int total = contacts.size();                // Total number of contacts
        int chunk = (total + size - 1) / size;      // Divide contacts across processes (ceil division)

        // Send chunks to all other worker processes
        for (int i = 1; i < size; i++) {
            string text = vector_to_string(contacts, i * chunk, (i + 1) * chunk);
            send_string(text, i);
        }

        // Process the first chunk of data locally
        start = MPI_Wtime(); // Start timing
        string result;
        for (int i = 0; i < min(chunk, total); i++) {
            string match = check(contacts[i], search_term);
            if (!match.empty()) result += match;
        }
        end = MPI_Wtime(); // End timing

        // Collect results from all worker processes
        for (int i = 1; i < size; i++) {
            string recv = receive_string(i);
            if (!recv.empty()) result += recv;
        }

        // Write all results to output file
        ofstream out("output.txt");
        out << result;
        out.close();

        // Print execution time
        printf("Process %d took %f seconds.\n", rank, end - start);

    } else {
        // Worker processes receive their chunk of data from master
        string recv_text = receive_string(0);
        vector<Contact> contacts = string_to_contacts(recv_text);

        // Process local chunk and search for matches
        start = MPI_Wtime();
        string result;
        for (auto &c : contacts) {
            string match = check(c, search_term);
            if (!match.empty()) result += match;
        }
        end = MPI_Wtime();

        // Send found matches back to master
        send_string(result, 0);
        printf("Process %d took %f seconds.\n", rank, end - start);
    }

    MPI_Finalize(); // Clean up and exit
    return 0;
}

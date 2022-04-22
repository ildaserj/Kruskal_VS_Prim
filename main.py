import random
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import numpy as np


class Graph:
    def __init__(self, num_of_nodes):
        self.m_num_of_nodes = num_of_nodes  # numero di nodi del grafo
        self.m_graph = []  # vettore di bordi: struttura dati usata per archiviare il grafo Kruskal
        self.m_matrix = []  # matrice di adiacenza per Prim

    # ogni bordo di qualsiasi grafo ponderato collega esattamente due nodi e ha un certo peso assegnato
    def set_number_of_nodes(self, num_nod):
        self.m_num_of_nodes = num_nod


    def add_edge(self, node1, node2, weight):  # aggiunge bordo
        self.m_graph.append([node1, node2, weight])

    def add_matrix(self, metrix):  # Prim
        self.m_matrix = metrix

    ########    KRUSKAL

    # ordinare elenco di bordi dal min al mag
    # inizializiamo e manteniamo due matrici ausiliarie di dim del numero di nodi nel grafo (una indica il nodo stesso l'altra il padre)
    # Kruskal costruisce dei sottoalberi che poi unisce prendendo il minor peso possibile
    # inizialmente si considera ogni singolo nodo come un albero separato e iniziamo a collegarli
    # quindi avremo i due array:: Nodes  --> |0|1|2|3|4|5|6|7|8|
    #                             Parents--> |0|1|2|3|4|5|6|7|8|    Ogni nodo è padre di se stesso
    # Scegliamo il bordo minimo e colleghiamo i nodi effettivamente es. 2-'1'-5 nodi 2 e 5 peso
    #                             Nodes  --> |0|1|2|3|4| 5 |6|7|8|
    #                             Parents--> |0|1|2|3|4|(2)|6|7|8|

    # Trova nodo radice di un sottoalbero contenente il nodo "i"
    # Ricerca ricorsiva, trova un sottoalbero che contiene il nodo i
    def find_subtree(self, parent, il):
        if parent[il] == il:
            return il
        return self.find_subtree(parent, parent[il])

    # Collega i sottoalberi contenenti i nodi 'x' e 'y'
    # trova i due sottoalberi, ne confronta le dim e collega il sottoalbero più piccolo a
    # quello più grande
    def connect_subtrees(self, parent, subtree_sizes, x, y):
        xroot = self.find_subtree(parent, x)
        yroot = self.find_subtree(parent, y)
        if subtree_sizes[xroot] < subtree_sizes[yroot]:
            parent[xroot] = yroot
        elif subtree_sizes[xroot] > subtree_sizes[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            subtree_sizes[xroot] += 1

    def kruskal_mst(self):
        time_start = timer()
        # Albero risultante
        result = []

        ip = 0
        e = 0  # Num di bordi in MST
        self.m_graph = sorted(self.m_graph, key=lambda item: item[2])  # ordina i bordi in base ai pesi

        parent = []
        subtree_size = []

        for node in range(self.m_num_of_nodes):
            parent.append(node)
            subtree_size.append(0)

        # il numero di bordi in un MST è uguale a (m_num_of_nodes -1)

        while e < (self.m_num_of_nodes - 1):  # scorre l'elenco di bordi, li sceglie uno per uno e se possibile li agiunge all'alb ris
            # scegli un bordo con il peso minimo
            node1, node2, weight = self.m_graph[ip]
            ip = ip + 1

            x = self.find_subtree(parent, node1)
            y = self.find_subtree(parent, node2)

            if x != y:
                e = e + 1
                result.append([node1, node2, weight])
                self.connect_subtrees(parent, subtree_size, x, y)
        time_end = timer()
        final_time = time_end - time_start
        for node1, node2, weight in result:
            print("%d - %d: %d" % (node1, node2, weight))

        print("Tempo impiegato Kruskal con ", self.m_num_of_nodes, " nodi: ", final_time)

        return final_time
    ####################################################################################
    #############                         PRIM                          ################

    def prim_mst(self):
        time = 0
        start_time = timer()
        # definiamo un numero grande, sarà sempre il peso più alto nei confronti
        positive_inf = float('inf')

        # lista, mostra quali nodi sono selezionati
        # così non selezioniamo 2 volte il solito, e sappiamo quando concludere il ciclo
        selected_nodes = [False for node in range(self.m_num_of_nodes)]

        # matrice dei risultati
        result = [[0 for column in range(self.m_num_of_nodes)]
                  for row in range(self.m_num_of_nodes)]
        index = 0

        # finche ci sono nodi che non sono insclusi in MST, continua a cercare
        while False in selected_nodes and timer():
            # usiamo il numero creato prima come peso minimo possibile
            minimum = positive_inf

            start = 0  # nodo iniziale
            end = 0  # nodo finale

            for i in range(self.m_num_of_nodes):
                # se il nodo £ MST, guarda i suoi archi
                if selected_nodes[i]:
                    for j in range(self.m_num_of_nodes):
                        # se il nodo ha un percorso verso il nodo finale E non incluso in MST (Evitare cicli)
                        if not selected_nodes[j] and self.m_matrix[i][j] > 0:
                            if self.m_matrix[i][j] < minimum:  # se il peso del percorso analizzato è < di minimum
                                minimum = self.m_matrix[i][j]  # impostiamolo come nuovo minimo, e ipmpostiamo il nuovo start e end
                                start, end = i, j
            # ultimo nodo al MST quindi già selezionato
            selected_nodes[end] = True

            # inserimento dati campi matrice adicacenza MST
            result[start][end] = minimum

            if minimum == positive_inf:
                result[start][end] = 0

            index += 1

            #result[end][start] = result[start][end]
            end_time = timer()
            time = end_time - start_time

        for i in range(len(result)):
            for j in range(0 + i, len(result)):
                if result[i][j] != 0:
                    print("%d - %d: %d" % (i, j, result[i][j]))
        print("tempo impiegato Prim ", self.m_num_of_nodes, " nodi: ", time)
        return time


def graph_creator(Dim, Prob, graph1, graph2):

    matrix = np.zeros((Dim, Dim))
    max_weight = 20
    for i in range(Dim):
        for j in range(i):
            if random.randint(1, 100) <= Prob and matrix[i][j] == 0:
                matrix[i][j] = random.randint(1, max_weight)
                matrix[j][j] = matrix[i][j]
                print(matrix)
                graph1.add_edge(i, j, matrix[i][j])
    graph2.m_matrix = matrix



if __name__ == '__main__':

    dim = 50

    TimeK = []
    TimeP = []

    graphP = Graph(dim)
    graphK = Graph(dim)
    probability = []
    dimen = []


    prob = 100

    while dim < 200: ###in range(20, 51):

        print(prob)
        probability.append(prob)
        dimen.append(dim)

        print("dimensione", dim, " Probabilità ", prob)
        for n in range(0, dim):

            matrix = np.zeros((dim, dim))
            max_weight = 20
            for i in range(dim):
                for j in range(i):
                    if(random.randint(1, 100)) <= prob and matrix[i][j] == 0:
                        matrix[i][j] = random.randint(1, max_weight)
                        matrix[j][i] = matrix[i][j] ########
                        graphK.add_edge(i, j, matrix[i][j])
                    graphP.m_matrix = matrix

        print(graphP.m_matrix)
        time1 = graphK.kruskal_mst()
        TimeK.append(time1)

        time2 = graphP.prim_mst()
        TimeP.append(time2)
        #prob += 10

        dim += 50
        graphK.set_number_of_nodes(dim)
        graphP.set_number_of_nodes(dim)


    plt.title("MST: Kruskal VS Prim")
    plt.plot(dimen, TimeK, label="MST: Kruskal")
    plt.plot(dimen, TimeP, label="MST: Prim")
    plt.legend()
    plt.xlabel("Numero nodi")############
    plt.ylabel("Tempo esecuzione")
    plt.show()



##











#   graph.add_edge(0, 1, 4)
#   graph.add_edge(0, 2, 7)
#   graph.add_edge(1, 2, 11)
#   graph.add_edge(1, 3, 9)
#   graph.add_edge(1, 5, 20)
#   graph.add_edge(2, 5, 1)
#   graph.add_edge(3, 6, 6)
#   graph.add_edge(3, 4, 2)
#   graph.add_edge(4, 6, 10)
#   graph.add_edge(4, 8, 15)
#   graph.add_edge(4, 7, 5)
#   graph.add_edge(4, 5, 1)
#   graph.add_edge(5, 7, 3)
#   graph.add_edge(6, 8, 5)
#   graph.add_edge(7, 8, 12)
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import random


class Grafo:
    def __init__(self, direcionado=False, ponderado=False):
        self.estrutura = {}
        self.direcionado = direcionado
        self.ponderado = ponderado
        self.N = 0

    def adiciona_vertice(self, u):
            try:
                self.estrutura[u]
            except KeyError:
                self.estrutura[u] = []
                self.N += 1

    def adiciona_aresta(self, u, v, peso=1):
        if self.ponderado:
            self.estrutura[u].append((v, peso))
            if not self.direcionado:
                self.estrutura[v].append((u, peso))
        else:
            self.estrutura[u].append(v)
            if not self.direcionado:
                self.estrutura[v].append(u)

    def remove_aresta(self, u, v):
        if v in self.estrutura[u]:
            self.estrutura[u].remove(v)

    def remove_vertice(self, u):
        if u in self.estrutura:
            del self.estrutura[u]
        for vertices in self.estrutura:
            if u in self.estrutura[vertices]:
                self.remove_aresta(vertices, u)

    def tem_aresta(self, u, v):
        for i in self.estrutura[u]:
            if not self.ponderado:
                if i == v:
                    return True
            else:
                nome = i[0]
                if nome == v:
                    return True
        return False

    def grau(self, u):
        return len(self.estrutura[u])

    def peso(self, u, v):
        for i in self.estrutura[u]:
            if not self.ponderado:
                if i == v:
                    return 1
            else:
                nome = i[0]
                if nome == v:
                    return i[1]
        return False

    def sao_conectados(self, u, v):
        for tupla in self.estrutura[v]:
            if u == tupla[0]:
                return True
        return False

    def imprime(self):
        for u in self.estrutura.items():
            print(u)

    def retorna_adjacentes(self, u):
        return self.estrutura[u]

    def minDist(self, cost, visited):
        maior = np.inf
        minIndex = None

        self.imprime()
        for v in self.estrutura:
            if minIndex is None:
                minIndex = v
            if cost[v][0] < maior and v not in visited:
                print(f'{cost[v][0]}   -   {maior} - V[{v}]')
                maior = cost[v][0]
                minIndex = v

        return None if minIndex is None else str(minIndex)

    @staticmethod
    def MinDistance(cost, visited):
        maior = np.inf
        print(cost)
        minIndex = 0
        for k, v in cost.items():
            print(f'{v[0]} < {maior}')
            if v[0] < maior and v not in visited:
                maior = v[0]
                minIndex = k
        return np.inf if maior == np.inf else minIndex

    def retorna_vizinhos(self, node):
        aux = []
        for i in self.retorna_adjacentes(node):
            aux.append(i[0])
        return aux

    @staticmethod
    def quantas_arestas(G):
        soma = 0
        for x in G.estrutura:
            soma += len(G.retorna_adjacentes(x))

        return soma

    def dijkstra(self, source_node):
        visited = []
        indices = []
        all_multiple_paths = []
        for x in self.estrutura:
            indices.append(x)

        values = [[np.inf, '-'] for _ in range(len(self.estrutura))]
        cost = {}
        for i in range(len(self.estrutura)):
            cost[indices[i]] = values[i]

        cost[source_node][0] = 0
        current_node = source_node
        while len(visited) < len(self.estrutura):
            print(f'LENVI: {len(visited)} LENEST: {len(self.estrutura)}')
            adjacent_nodes = self.retorna_adjacentes(current_node)
            for ind in range(len(adjacent_nodes)):
                if adjacent_nodes[ind][0] not in visited:
                    accumulated = 1 + cost[current_node][0]
                    if accumulated == cost[adjacent_nodes[ind][0]][0]:
                        multiple_path = [adjacent_nodes[ind], current_node]
                        atual = cost[current_node][1]
                        multiple_path.append(atual)

                        while atual != '-':
                            atual = cost[atual][1]
                            if atual != '-':
                                multiple_path.append(atual)

                        multiple_path.reverse()
                        all_multiple_paths.append(multiple_path)

                    if accumulated < cost[adjacent_nodes[ind][0]][0]:
                        cost[adjacent_nodes[ind][0]][0] = accumulated
                        cost[adjacent_nodes[ind][0]][1] = current_node

            visited.append(current_node)
            print(cost)
            # aux = self.minDist(cost, visited)

            current_node = self.minDist(cost, visited)
            if current_node is None:
                print('\n\n\n\n\nentrou')
                break
            print(current_node)

        result = []
        for x in cost.values():
            if x[0] != np.inf:
                if x[0] != 0 and x[1] != 0:
                    result.append(x[0])
        print(cost)
        return cost, result, all_multiple_paths

    # em implementacao
    def kruskal(self):
        tree = Grafo(direcionado=False, ponderado=True)
        indices = []

        # cria vertices
        for x in self.estrutura:
            tree.adiciona_vertice(x)

        for x in self.estrutura:
            indices.append(x)

        # pega vertice aleatorio como comeco

        ordenado_aux = {}
        ordenado = {}

        for k, v in self.estrutura.items():
            for i in v:
                ordenado_aux[i[1]] = [i[0], k]
        # print(ordenado_aux)

        for key in sorted(ordenado_aux.keys()):
            ordenado[key] = ordenado_aux[key]

        lista_keys = []
        for k in ordenado.keys():
            lista_keys.append(k)
        contador = 0

        while self.quantas_arestas(tree) != len(tree.estrutura) - 1:

            # tree.imprime()
            if contador == len(lista_keys) - 1:
                contador = 0
            k = lista_keys[contador]

            tree.adiciona_aresta(ordenado[k][0], ordenado[k][1], k)

            contador += 1

    @property
    def prim(self):
        visited = []
        indices = []
        total_cost = 0
        for x in self.estrutura:
            indices.append(x)

        # pega vertice aleatorio como comeco
        source_node = random.choice(indices)
        values = [[np.inf, '-'] for _ in range(len(self.estrutura))]
        cost = {}
        for i in range(len(self.estrutura)):
            cost[indices[i]] = values[i]

        cost[source_node][0] = 0
        current_node = source_node
        while len(visited) < len(self.estrutura):
            adjacent_nodes = self.retorna_adjacentes(current_node)
            for ind in range(len(adjacent_nodes)):
                if adjacent_nodes[ind][0] not in visited:
                    accumulated = self.estrutura[current_node][ind][1]
                    if accumulated < cost[adjacent_nodes[ind][0]][0]:
                        cost[adjacent_nodes[ind][0]][0] = accumulated
                        cost[adjacent_nodes[ind][0]][1] = current_node

            visited.append(current_node)
            current_node = self.minDist(cost, visited)

        tree = Grafo(direcionado=False, ponderado=True)

        # cria vertices
        for x in cost:
            tree.adiciona_vertice(x)

        # adiciono arestas
        for v, k in cost.items():
            tree.adiciona_aresta(v, k[1], k[0])
            total_cost += k[0]

        return tree, total_cost

    def DFS_interative(self, node, interest=None):
        visited = []
        stack = [node]

        while stack:
            s = stack.pop()

            if s not in visited:
                visited.append(s)

                stack += [x[0] for x in self.estrutura[s][::-1] if x not in visited]
            if s == interest:
                return visited
        return visited

    def dfs_numberOf(self, node, visited):

        visited.append(node)
        for i in self.retorna_adjacentes(node):
            if i not in visited:
                visited = self.dfs_numberOf(i, visited)

        return visited

    def numberOfComponents(self):
        visited = []
        count = 0
        for x in self.estrutura:
            if x not in visited:
                visited = self.dfs_numberOf(x, visited)
                count += 1

        return count

    def random_graph_NM(self, n, m):

        for i in range(n):
            self.adiciona_vertice(f'{i}')

        while self.size_arestas() < m:
            x, y = np.random.randint(n), np.random.randint(n)
            if x != y and not self.tem_aresta(f'{x}', f'{y}'):
                if x not in self.retorna_adjacentes(f'{y}'):
                    self.adiciona_aresta(f'{x}', f'{y}')

    def size_arestas(self):
        soma = 0
        for x in self.estrutura:
            soma += len(self.retorna_adjacentes(x))
        return soma / 2

    def get_all_degrees(self):
        degrees = []
        for i in self.estrutura:
            degrees.append(self.grau(i))

        return degrees

    def eccentricity(self, u):
        lista, menores_caminhos, path, list_path = self.dijsktra2(u)
        print(f'Menores: {u}, {lista}')
        return max(menores_caminhos)

    def diameter(self):
        maiores = []
        for node in self.estrutura:
            maiores.append(self.eccentricity(node))

        # print(maiores)
        return max(maiores)

    def radius(self):
        maiores = []
        count = 1
        for node in self.estrutura:
            print(f'to em: {node}')
            print(f'N: {count}')
            maiores.append(self.eccentricity(node))
            count += 1

        print(maiores)
        return min(maiores)

    def coef_local(self, u):

        adj = self.retorna_adjacentes(u)
        lista_conectados = []

        for x in adj:
            for y in adj:
                if x in self.retorna_adjacentes(y):
                    if [x, y] not in lista_conectados and [y, x] not in lista_conectados:
                        lista_conectados.append([x, y])

        degree = self.grau(u)
        if degree == 1:
            return 1
        elif degree > 1:
            result = 2 * len(lista_conectados) / (degree * (degree - 1))
            return result
        else:
            return 0

    # Average Clustering Coefficient:
    def coef_local_medio(self):

        soma = 0
        for x in self.estrutura:
            coef_local = self.coef_local(x)
            print(f"Coeficiente de Agrupamento Local de {x}: {coef_local}")
            soma += coef_local

        result = (1 / self.N) * soma
        return result

    def menores_caminhos(self):
        menores_caminhos_possiveis = []
        for i in self.estrutura:
            print(i)
            aux = self.dijkstra(i)[1]
            menores_caminhos_possiveis += aux

        return menores_caminhos_possiveis

    def closeness_centrality(self):
        maior = 0
        vertice = ""
        N = len(self.estrutura)
        for x in self.estrutura:
            closeness = (N - 1) / sum(self.dijkstra(x)[1])
            if closeness > maior:
                maior = closeness
                vertice = x

        return vertice, maior

    def betweenness_centrality(self, target):

        bet = {}
        list_paths = []
        for node in self.estrutura:
            if node != target:
                lista, not_used, aux = self.dijkstra(node)

                all_multiple_paths = []
                for path in aux:
                    if path[-1] != target and path[0] == node:
                        all_multiple_paths.append(path)

                cost = {}
                for k, v in lista.items():
                    if v[1] != '-' and k != target:
                        cost[k] = v

                for k, v in cost.items():
                    shortest_path = [k]
                    atual = lista[k][1]
                    shortest_path.append(atual)

                    while atual != '-':
                        atual = lista[atual][1]
                        if atual != '-':
                            shortest_path.append(atual)
                    # print(shortest_path)
                    shortest_path_principal = shortest_path.copy()
                    shortest_path.reverse()

                    aux = shortest_path.copy()
                    aux.sort()
                    if aux not in list_paths and shortest_path not in list_paths and shortest_path_principal not in list_paths: # noqa
                        list_paths.append(shortest_path)
                aux = []
                for x in all_multiple_paths:
                    a = x.copy()
                    aux.append(a)

                reverse_multiple = []

                for x in aux.copy():
                    a = x.copy()
                    a.reverse()
                    reverse_multiple.append(a)
                for x in range(len(aux)):
                    if aux[x] not in list_paths and reverse_multiple[x] not in list_paths and all_multiple_paths[x] not in list_paths: # noqa
                        list_paths = list_paths + [all_multiple_paths[x]]

        for path in list_paths:
            pair = [path[0], path[-1]]
            pair.sort()
            if pair[0] + pair[1] not in bet.keys():
                if target in path:
                    bet[pair[0] + pair[1]] = [1, 1]
                else:
                    bet[pair[0] + pair[1]] = [0, 1]
            else:
                if target in path:
                    bet[pair[0] + pair[1]][0] = bet[pair[0] + pair[1]][0] + 1
                bet[pair[0] + pair[1]][1] = bet[pair[0] + pair[1]][1] + 1

        calculo = 0
        for v in bet.values():
            calculo += v[0] / v[1]

        return calculo / (((self.N - 1) * (self.N - 2)) / 2)

        # 15  20

    def scale_free_model(self, n, m):
        self.adiciona_vertice('1')
        self.adiciona_vertice('2')
        self.adiciona_vertice('3')
        self.adiciona_vertice('4')
        self.adiciona_vertice('5')

        self.adiciona_aresta('1', '2')
        self.adiciona_aresta('2', '3')
        self.adiciona_aresta('2', '4')
        self.adiciona_aresta('3', '5')
        self.adiciona_aresta('3', '4')
        self.adiciona_aresta('4', '5')

        while self.N < n:
            print(f'CONT: {self.N}')
            self.barabasi(G, n, 2)

    def check_probability(self):
        p = {}
        degrees = {aux: self.grau(aux) for aux in self.estrutura}
        last = list(degrees.keys())[-1]
        degrees.pop(last)
        for node in self.estrutura:
            if node != str(self.N):
                p[node] = float(degrees[node]) / sum(degrees.values())

        node_probabilities = {}
        prev = 0
        for n, px in p.items():
            node_probabilities[n] = prev + px
            prev += px

        # print(node_probabilities)
        return node_probabilities

    @staticmethod
    def barabasi(G, n0, k=2):
        contador = 0

        while contador < n0:
            k_contador = 0
            G.adiciona_vertice(f"{G.N + 1}")

            p_all = G.check_probability()
            aux = p_all.copy()
            linkeds = []
            while k_contador < k:

                number = random.random()

                key = list(aux.keys())[0]
                p = aux.pop(key)

                if number < p:
                    G.adiciona_aresta(f"{G.N}", key)
                    linkeds.append(key)
                    aux = p_all.copy()
                    # delete nodes that already has connection to last node
                    for linked in linkeds:
                        aux.pop(linked)
                    k_contador += 1
            contador += 1

    def maior_betweeneess_centrality(self):
        lista = {}
        for target in self.estrutura:
            lista[target] = self.betweenness_centrality(target)

        print(lista)

    def dijsktra2(self, initial):
        visited = {initial: 0}
        path = {}
        nodes = set(self.estrutura)

        while nodes:
            min_node = None
            for node in nodes:
                try:
                    visited[node]
                except KeyError:
                    continue
                if min_node is None:
                    min_node = node
                elif visited[node] < visited[min_node]:
                    min_node = node

            if min_node is None:
                break
            nodes.remove(min_node)
            current_weight = visited[min_node]

            for edge in self.retorna_adjacentes(min_node):
                if self.ponderado:
                    weight = current_weight + self.peso(min_node, edge)
                else:
                    weight = current_weight + 1
                try:
                    visited[edge]
                except KeyError:
                    visited[edge] = weight
                    path[edge] = min_node
                    continue
                if weight < visited[edge]:
                    visited[edge] = weight
                    path[edge] = min_node
        # print(visited)
        # print(path)
        list_path = {node: [value] for node, value in path.items()}

        for node, value in path.items():
            if value != initial:
                target = value
                while target != initial:
                    list_path[node].append(path[target])
                    target = path[target]

        # print(list_path)
        result = [x for x in visited.values() if x != 0]
        return visited, result, path, list_path

    def mean_degree(self):
        return sum(self.get_all_degrees()) / len(self.get_all_degrees())

    def write_pajek_file(self):
        f = open("barabasi_pajek.net", "w")

        with f as line:
            line.write(f'*Vertices  {self.N} \n')
            count = 1
            line_key = {}
            for vertice in self.estrutura:
                line.write(f'{count} "{vertice}"\n')
                line_key[vertice] = count
                count += 1

            line.write(f'*Arcs\n')
            line.write(f'*Edges\n')
            for vertice in self.estrutura:
                for adjacent in self.retorna_adjacentes(vertice):
                    line.write(f'{str(line_key[vertice])} {(line_key[adjacent])} {self.peso(vertice, adjacent)}\n')

    def read_pajek_file(self, file):
        path = open(file, "r")
        isVertices = False
        isEdges = False
        isArcs = False
        dict_index = {}
        try:
            with path as f:
                line = f.readline()
                while line:
                    split = line.split()
                    # if len(split) > 0:
                    if split[0] == '*Vertices':
                        # print(split[0].upper())
                        line = f.readline()
                        split = line.split()
                        isVertices = True
                    if split[0].upper() == '*ARCS' or split[0].upper() == '*ARCSLIST':
                        isArcs = True
                        isVertices = False
                        line = f.readline()
                        split = line.split()
                    if split[0].upper() == '*EDGES' or split[0].upper() == '*EDGESLIST':
                        isEdges = True
                        isArcs = False
                        line = f.readline()

                    if isVertices and not isEdges and not isArcs:
                        vertice_split = line.split()
                        node = ""

                        for word in vertice_split[1:]:
                            node += word

                        aux = node.split('"')
                        node = ""
                        for word in aux:
                            node += word
                        dict_index[vertice_split[0]] = node
                        self.adiciona_vertice(node)
                    elif isEdges and not isVertices and not isArcs:
                        edges_split = line.split()

                        for edge in edges_split[1:]:
                            self.adiciona_aresta(dict_index[edges_split[0]], dict_index[edge])

                    line = f.readline()

                    # if isVertices:
                    # print(line.split())
                    # self.adiciona_vertice(line)

        except IOError:
            return ''
        pass

    def is_conexo(self):
        if self.direcionado:
            return self.isSC()
        dfs = self.DFS_interative(list(self.estrutura.keys())[0])
        return len(dfs) == len(self.estrutura)

    def getTranspose(self):
        g = Grafo(direcionado=self.direcionado, ponderado=self.ponderado)
        for i in self.estrutura:
            for j in self.estrutura[i]:
                g.adiciona_vertice(j)
                g.adiciona_aresta(j, i)
        return g

    def isSC(self):
        visited = set(self.DFS_interative(list(self.estrutura)[0])) ^ set(list(self.estrutura))
        if len(visited) > 0:
            return False
        gr = self.getTranspose()

        visited = set(gr.DFS_interative(list(gr.estrutura)[0])) ^ set(list(gr.estrutura))
        if len(visited) > 0:
            return False
        return True

    def is_euleriano(self):
        if not self.direcionado:
            impares = len([x for x in self.estrutura.values() if len(x) % 2 != 0])
            return impares <= 2 and self.is_conexo()
        else:
            pass


G = Grafo(direcionado=False, ponderado=False)

G.read_pajek_file('erdos.net')

print(f'{G.is_euleriano()}')

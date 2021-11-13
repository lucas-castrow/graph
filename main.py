from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import random

class Grafo:
    def __init__(self, direcionado=False, ponderado=False):
        self.estrutura = defaultdict(list)
        self.direcionado = direcionado
        self.ponderado = ponderado

    def adiciona_vertice(self, u):
        if u not in self.estrutura:
            self.estrutura[u] = []

    def adiciona_aresta(self, u, v, peso=1):
        if self.ponderado:
            self.estrutura[u].append((v, peso))
            if not self.direcionado:
                self.estrutura[v].append((u, peso))
        else:
            self.estrutura[u].append((v))
            if not self.direcionado:
                self.estrutura[v].append((u))

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
            nome = i[0]
            if nome == v:
                peso = i[1]
                print(peso)
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
        minIndex = 0

        for v in self.estrutura:
            if cost[v][0] < maior and v not in visited:
                maior = cost[v][0]
                minIndex = v
        return str(minIndex)

    def retorna_vizinhos(self, node):
        aux = []
        for i in self.retorna_adjacentes(node):
            aux.append(i[0])
        return aux

    def quantas_arestas(self, G):
        soma = 0
        for x in G.estrutura:
            soma += len(G.retorna_adjacentes(x))

        return soma

    def dijkstra(self, source_node):
        visited = []
        indices = []
        for x in self.estrutura:
            indices.append(x)

        values = [[np.inf, '-'] for i in range(len(self.estrutura))]
        cost = {}
        for i in range(len(self.estrutura)):
            cost[indices[i]] = values[i]

        cost[source_node][0] = 0
        current_node = source_node
        while len(visited) < len(self.estrutura):
            adjacent_nodes = self.retorna_adjacentes(current_node)
            for ind in range(len(adjacent_nodes)):
                if adjacent_nodes[ind][0] not in visited:
                    accumulated = 1 + cost[current_node][0]
                    if accumulated < cost[adjacent_nodes[ind][0]][0]:
                        cost[adjacent_nodes[ind][0]][0] = accumulated
                        cost[adjacent_nodes[ind][0]][1] = current_node

            visited.append(current_node)
            current_node = self.minDist(cost, visited)

        result = []
        print(cost)
        for x in cost.values():
            if x[0] != np.inf:
                if x[0] != 0 and x[1] != 0:
                    result.append(x[0])

        return cost,result

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

    def prim(self):
        visited = []
        indices = []
        total_cost = 0
        for x in self.estrutura:
            indices.append(x)

        # pega vertice aleatorio como comeco
        source_node = random.choice(indices)
        values = [[np.inf, '-'] for i in range(len(self.estrutura))]
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

                for x in self.estrutura[s][::-1]:
                    if x not in visited:
                        stack.append(x[0])
            if s == interest:
                return visited
        return visited

    def dfs_tj(self, node, visited, result, visiting, count, trees):
        flag = False

        if node not in visited:
            res = []
            visiting.append(node)

            for x in self.estrutura[node][::-1]:
                visited, result, visiting, count, trees = self.dfs_tj(x, visited, result, visiting, count, trees)
                for i in self.retorna_vizinhos(node):
                    if i in visiting:
                        flag = True
                        count += 1

                if flag:
                    [res.append(x) for x in visiting if x not in res]
                    trees.append(res)

                    break

            visiting.remove(node)
            visited.append(node)
            if self.retorna_adjacentes(node) == []:
                trees.append([node])
            result.append(node)

        return visited, result, visiting, count, trees

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

    def isCyclic(self):
        # pego primeiro vertice
        first = next(iter(self.estrutura))
        count = self.tarjan(first)[3]
        if count > 0:
            return True
        else:
            return False

    def scc(self, node):

        aux = self.tarjan(node)[4]

        resposta = []
        tree = []
        for i in aux:
            tree.append(tuple(sorted(i)))

        result = list(set(tree))
        for i in result:
            for x in result:
                if x != i:
                    if len(i) > 1 and len(x) > 1:
                        if len(x) > len(i):
                            dif = tuple(set(x) - set(i))
                            if i not in resposta:
                                resposta.append(i)
                            resposta.append(dif)
                    if len(i) == 1:
                        if i not in resposta:
                            resposta.append(i)

        return resposta

    def tarjan(self, node, interest=None):
        result = []
        visited = []
        unvisited = []
        visiting = []
        trees = []
        count = 0
        # coloco todos vertices nao visitados
        for x in self.estrutura:
            unvisited.append(x)

        # enquanto tiver vertice em nao-visitados
        while unvisited:
            # quando acaba o dfs de cada vertice, chamo outro nao visitado ainda
            visited, result, visiting, count, trees = self.dfs_tj(unvisited[0], visited, result, visiting, count, trees)
            # faco diferenca de visitados com nao-visitados
            unvisited = list(set(unvisited) - set(visited))

        # inverto para ficar da ordem
        result.reverse()
        return visited, result, visiting, count, trees

    def random_graph_NM(self, n, m):

        for i in range(n):
            self.adiciona_vertice(f'{i}')

        while self.size_arestas() < m:
            x, y = np.random.randint(n), np.random.randint(n)
            if x != y and not self.tem_aresta(f'{x}',f'{y}'):
                if x not in self.retorna_adjacentes(f'{y}'):
                    self.adiciona_aresta(f'{x}',f'{y}')

    def size_arestas(self):
        soma = 0
        for x in self.estrutura:
            soma += len(G.retorna_adjacentes(x))
        return soma

    def get_all_degrees(self):
        degrees = []
        for i in self.estrutura:
            degrees.append(self.grau(i))

        return degrees

    def eccentricity(self,u):
        lista = self.dijkstra(u)[0]
        menores_caminhos = [k[0] for k in lista.values()]
        print(max(menores_caminhos))
        return max(menores_caminhos)

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


    def coef_local_medio(self):
        N = len(self.estrutura)

        soma = 0
        for x in self.estrutura:
            coef_local = self.coef_local(x)
            print(f"Coeficiente de Agrupamento Local de {x}: {coef_local}")
            soma += coef_local

        result = (1/N) * soma
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
            closeness = (N-1) / sum(self.dijkstra(x)[1])
            if closeness > maior:
                maior = closeness
                vertice = x

        return vertice, maior

    def betweenness_centrality(self):
        pass
G = Grafo(direcionado= False, ponderado = False)

# G.adiciona_vertice('A')
# G.adiciona_vertice('B')
# G.adiciona_vertice('C')
# G.adiciona_vertice('D')
# G.adiciona_vertice('E')
# G.adiciona_vertice('F')
# G.adiciona_vertice('G')
#
# G.adiciona_aresta('A','D')
# G.adiciona_aresta('A','C')
# G.adiciona_aresta('A','F')
#
# G.adiciona_aresta('B','E')
# G.adiciona_aresta('B','F')
#
# G.adiciona_aresta('C','G')
# G.adiciona_aresta('C','D')
# G.adiciona_aresta('C','E')
#
# G.adiciona_aresta('D','C')
#
# G.adiciona_aresta('E','F')

G.random_graph_NM(2000,5000)
#G.imprime()
#print(f"Coeficiente de Agrupamento Médio: {G.coef_local_medio()}")
#print(len(G.estrutura))
#print(f"arestas = {G.size_arestas()}")


#G.eccentricity('D')
#------------- PRINTAR GET ALL DEGRES (0.5) pontos
#get_all = (G.get_all_degrees())
# plt.hist(get_all)
# plt.xlabel("")
# plt.ylabel("")
# plt.show()
menores = G.menores_caminhos()
plt.hist(menores)
plt.xlabel("")
plt.ylabel("")
plt.show()


#print(G.closeness_centrality())
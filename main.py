from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import random

class Grafo:
    def __init__(self, direcionado=False, ponderado=False):
        self.estrutura = defaultdict(list)
        self.direcionado = direcionado
        self.ponderado = ponderado
        self.N = 0

    def adiciona_vertice(self, u):
        if u not in self.estrutura:
            self.estrutura[u] = []
            self.N += 1

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
        minIndex = 0

        self.imprime()
        for v in self.estrutura:
            if cost[v][0] < maior and v not in visited:
                print(f'{cost[v][0]}   -   {maior} - V[{v}]')
                maior = cost[v][0]
                minIndex = v


        return str(minIndex)

    def MinDistance(self, cost, visited):
        maior = np.inf
        print(cost)

        for k,v in cost.items():
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

    def quantas_arestas(self, G):
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

        values = [[np.inf, '-'] for i in range(len(self.estrutura))]
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
                        multiple_path = [adjacent_nodes[ind]]
                        multiple_path.append(current_node)
                        atual = cost[current_node][1]
                        multiple_path.append(atual)

                        while(atual != '-'):
                            atual = cost[atual][1]
                            if atual != '-':
                                multiple_path.append(atual)

                        multiple_path.reverse()
                        all_multiple_paths.append(multiple_path)
                    # if ind not in cost:
                    #     cost[adjacent_nodes[ind][0]][0] = accumulated
                    #     cost[adjacent_nodes[ind][0]][1] = current_node
                    if accumulated < cost[adjacent_nodes[ind][0]][0]:
                        cost[adjacent_nodes[ind][0]][0] = accumulated
                        cost[adjacent_nodes[ind][0]][1] = current_node


            visited.append(current_node)
            print(cost)
            #aux = self.minDist(cost, visited)
            current_node = self.minDist(cost, visited)

            print(current_node)

        result = []
        for x in cost.values():
            if x[0] != np.inf:
                if x[0] != 0 and x[1] != 0:
                    result.append(x[0])

        return cost,result, all_multiple_paths

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
            soma += len(self.retorna_adjacentes(x))
        return soma/2

    def get_all_degrees(self):
        degrees = []
        for i in self.estrutura:
            degrees.append(self.grau(i))

        return degrees

    def eccentricity(self,u):
        lista = self.dijsktra2(u)[1]
        #menores_caminhos = [k[0] for k in lista.values()]
        #print(max(lista))
        return max(lista)

    def diameter(self):
        maiores = []
        for node in self.estrutura:
            maiores.append(self.eccentricity(node))

        print(maiores)
        return max(maiores)

    def radius(self):
        maiores = []
        for node in self.estrutura:
            maiores.append(self.eccentricity(node))

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

                for k,v in cost.items():
                    shortest_path = [k]
                    atual = lista[k][1]
                    shortest_path.append(atual)

                    while(atual != '-'):
                        atual = lista[atual][1]
                        if atual != '-':
                            shortest_path.append(atual)
                   # print(shortest_path)
                    shortest_path_principal = shortest_path.copy()
                    shortest_path.reverse()

                    aux = shortest_path.copy()
                    aux.sort()
                    if aux not in list_paths and shortest_path not in list_paths and shortest_path_principal not in list_paths:
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
                    if aux[x] not in list_paths and reverse_multiple[x] not in list_paths and all_multiple_paths[x] not in list_paths:
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
            calculo += v[0]/v[1]

        return calculo/(((self.N-1)*(self.N-2))/2)

                            #15  20
    def scale_free_model(self, n, m):
        G = Grafo(direcionado=False, ponderado=False)

        G.adiciona_vertice('1')
        G.adiciona_vertice('2')
        G.adiciona_vertice('3')
        G.adiciona_vertice('4')
        G.adiciona_vertice('5')

        G.adiciona_aresta('1', '2')
        G.adiciona_aresta('2', '3')
        G.adiciona_aresta('2', '4')
        G.adiciona_aresta('3', '5')
        G.adiciona_aresta('3', '4')
        G.adiciona_aresta('4', '5')

        while G.N < n:
            print(f'CONT: {G.N}')
            G.barabasi(G, n, 2)

        return G

    def check_probability(self):
        p = {}
        degrees = {aux: self.grau(aux) for aux in self.estrutura}
        last = list(degrees.keys())[-1]
        degrees.pop(last)
        for node in self.estrutura:
            if node != str(self.N):
                p[node] = float (degrees[node]) / sum(degrees.values())

        node_probabilities = {}
        prev = 0
        for n, px in p.items():
            node_probabilities[n] = prev+px
            prev += px

        #print(node_probabilities)
        return node_probabilities


    def barabasi(self, G, n0, k=2):
        contador = 0

        while contador < n0:
            k_contador = 0
            G.adiciona_vertice(f"{G.N+1}")

            p_all = G.check_probability()
            #p_all = dict(sorted(p_all.items(), key=lambda item: item[1], reverse=True))
            while k_contador < k:

                    number = random.random()
                    key = list(p_all.keys())[0]
                    p = p_all.pop(key)

                    if number < p:
                        G.adiciona_aresta(f"{G.N}",key)
                        k_contador += 1
            contador += 1


    def maior_betweeneess_centrality(self):
        lista = {}
        for target in self.estrutura:
            lista[target] = self.betweenness_centrality(target)

        print(lista)

    def read_pajek_files(self):
        pass

    def dijsktra2(self, initial):
        visited = {initial: 0}
        path = {}

        nodes = set(self.estrutura)

        while nodes:
            min_node = None
            for node in nodes:
                if node in visited:
                    if min_node is None:
                        min_node = node
                    elif visited[node] < visited[min_node]:
                        min_node = node

            if min_node is None:
                break

            nodes.remove(min_node)
            current_weight = visited[min_node]

            for edge in self.retorna_adjacentes(min_node):
                weight = current_weight + 1
                if edge not in visited or weight < visited[edge]:
                    visited[edge] = weight
                    path[edge] = min_node

            result = []
            print(visited)
            for x in visited.values():
                if x != 0:
                    result.append(x)
        return visited, result, path
    def write_pajek_file(self):
        f = open("barabasi_pajek.net", "w")

        with f as line:
            line.write(f'*Vertices  {self.N} \n')
            count = 1
            line_key = {}
            for vertice in self.estrutura:
                line.write(f'{count} {vertice}\n')
                line_key[vertice] = count
                count += 1

            line.write(f'*Arcs\n')
            line.write(f'*Edges\n')
            for vertice in self.estrutura:
                for adjacent in self.retorna_adjacentes(vertice):
                    line.write(f'{str(line_key[vertice])} {(line_key[adjacent])} {self.peso(vertice, adjacent)}\n')

G = Grafo(direcionado= False, ponderado = False)


X = G.scale_free_model(100,500)
X.imprime()
print(X.radius())
print(X.diameter())
print(f'DEGREES {sum(X.get_all_degrees())/len(X.get_all_degrees())}')

#print(sum(X.get_all_degrees())/len((X.get_all_degrees())))

#X.write_pajek_file()

#X = G.scale_free_model(5000,10000)

#X.imprime()
#print(X.size_arestas())


#G.random_graph_NM(5000,10000)

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
# menores = G.menores_caminhos()
# plt.hist(menores)
# plt.xlabel("")
# plt.ylabel("")
# plt.show()

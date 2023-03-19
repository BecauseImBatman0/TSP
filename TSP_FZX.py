import csv
import matplotlib.pyplot as plt
import numpy as np
import math


class Particle:
    def __init__(self, path):
        # 当前路径
        self.path = path
        # 当前距离
        self.dist = self.calculate_len(path)
        # 历史最短路径
        self.pbest = path
        # 历史最短路径的距离
        self.pbest_dist = self.calculate_len(path)

    # 计算一个粒子的路径长度
    def calculate_len(self, path):
        first = path[0]
        last = path[-1]
        dist = MATRIX[first][last]
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            dist += MATRIX[a][b]
        return dist

    # sigmoid函数
    def sigmoid(self, k, x):
        return k / (1 + np.exp(-x))

    # 更新粒子
    def update(self, alpha, best):
        new1 = self.path.copy()
        new2 = new1
        # 迭代次数为[0,4]
        iteration = int(self.sigmoid(4, alpha))

        l1 = 0
        l2 = 0
        for i in range(iteration):
            cities = [t for t in range(len(self.path))]
            t = np.random.choice(cities, 2)
            x = min(t)
            y = max(t)
            cross_part = best[x:y]
            tmp = []
            for t in new1:
                if t in cross_part:
                    continue
                tmp.append(t)
            # 两种交叉方法
            new1 = tmp + cross_part
            l1 = self.calculate_len(new1)
            new2 = cross_part + tmp
            l2 = self.calculate_len(new2)
        if l1 < l2:
            self.path = new1
            return l1
        else:
            self.path = new2
            return l2

    # 设置丢弃率
    def dropout(self, beta):
        rate = self.sigmoid(1, beta)
        new = self.path.copy()

        if np.random.rand() < rate:
            cities = [t for t in range(len(self.path))]
            t = np.random.choice(cities, 2)
            x, y = min(t), max(t)
            new[x], new[y] = new[y], new[x]
        l2 = self.calculate_len(new)
        return l2


class Model:
    def __init__(self, alpha, beta, iteration=200, num_particles=100):
        # 迭代次数
        self.iteration = iteration
        # 粒子数目
        self.num_particles = num_particles
        # 速度参数
        self.alpha = alpha
        self.beta = beta
        # 城市个数
        self.num_cities = len(MATRIX)
        # 粒子群
        self.particles = self.init_particles()

    # 随机初始化
    def random_init(self, num_particles):
        tmp = [x for x in range(self.num_cities)]
        result = []
        for i in range(num_particles):
            np.random.shuffle(tmp)
            particle = Particle(tmp.copy())
            result.append(particle)

        return result

    # 使用贪心算法初始化种群
    def init_particles(self):
        begin = 0
        result = []

        for i in range(self.num_particles):
            rest = [x for x in range(0, self.num_cities)]
            # 所有起始点都已经生成了
            if begin >= self.num_cities:
                # 所有起始点都已经生成了之后，剩下的结点随机生成，这是第二次优化
                others = self.random_init(self.num_particles - begin)
                result += others
                break
                # begin = np.random.randint(0, self.num_cities)
                # result.append(result[begin])
                # continue
            current = begin
            rest.remove(current)
            # 找到一条最近邻路径
            result_one = [current]
            while len(rest) != 0:
                tmp_min = math.inf
                tmp_choose = -1
                for x in rest:
                    if MATRIX[current][x] < tmp_min:
                        tmp_min = MATRIX[current][x]
                        tmp_choose = x
                current = tmp_choose
                result_one.append(tmp_choose)
                rest.remove(tmp_choose)
            particle = Particle(result_one)
            result.append(particle)
            begin += 1
        return result

    # 得到全局最优解
    def get_gbest(self):
        gbest = self.particles[0]
        for particle in self.particles:
            if particle.pbest_dist < gbest.pbest_dist:
                gbest = particle
        return gbest

    # 迭代
    def run(self):
        for epoch in range(self.iteration):
            # 更新全局最优解
            self.gbest = self.get_gbest()
            # 对每个粒子
            for particle in self.particles:
                pbest_dist = particle.pbest_dist
                # 与个体最优解交叉
                new_l1 = particle.update(self.alpha, particle.pbest)

                if new_l1 < pbest_dist:
                    particle.pbest = particle.path
                    particle.pbest_dist = new_l1

                # 与全局最优解交叉
                new_l2 = particle.update(self.alpha, self.gbest.pbest)

                if new_l2 < pbest_dist:
                    particle.pbest = particle.path
                    particle.pbest_dist = new_l2

                # 丢弃
                new_l3 = particle.dropout(self.beta)

                if new_l3 < pbest_dist:
                    particle.pbest = particle.path
                    particle.pbest_dist = new_l3

            print("第", epoch + 1, "次", self.gbest.pbest, "|距离|", self.gbest.pbest_dist)


# 读取csv文件
def read_cities(path):
    f = open(path, encoding='UTF8')
    reader = csv.reader(f)

    cities = []
    count = 0
    for line in reader:
        count += 1
        if count == 1:
            continue
        city = [float(line[1]), float(line[2])]
        cities.append(city)

    cities = np.array(cities)
    return cities


# 计算城市间距离
def get_MATRIX(num_cities, coordinates):
    matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i == j:
                matrix[i][j] = np.inf
                continue
            a = coordinates[i]
            b = coordinates[j]
            tmp = np.sqrt(sum([(x[0] - x[1]) ** 2 for x in zip(a, b)]))
            matrix[i][j] = tmp
    return matrix


# 坐标映射
def getCoordinates(path, data):
    coordinate = {}
    for i, city in enumerate(data):
        coordinate[i] = data[i]

    cors = []
    for i in path:
        if i in coordinate:
            cors.append(coordinate[i])
    cors = np.array(cors)
    return cors


# 显示路径
def show_image(gbest):
    path = gbest.pbest
    cors = getCoordinates(path, cities)
    print("TSP最终结果", path, "|距离|", gbest.pbest_dist)
    cors = np.vstack([cors, cors[0]])
    plt.scatter(cors[:, 0], cors[:, 1])
    cors = np.vstack([cors, cors[0]])
    plt.plot(cors[:, 0], cors[:, 1], "r")

    plt.show()


# 炼丹用
def train(epochs,batch_size):
    f = open("parameters.txt", "a")
    for epoch in range(epochs):
        print("epoch",epoch)
        alpha = 4*np.random.rand()
        beta = 2*np.random.rand()
        iteration = 300
        num_particles = int(50 + 300 * np.random.rand())
        for j in range(batch_size):
            print("epoch",epoch,"   batch",j)
            model = Model(alpha=alpha, beta=beta, iteration=iteration, num_particles=num_particles)
            model.run()
            x = model.gbest.pbest_dist
            s = ""
            for i in [x, alpha, beta, iteration, num_particles]:
                s += str(i) + " "
            f.write(s)
            f.write('\n')
    f.close()

# 验证用
def test(alpha, beta,iteration,num_particles,epochs,path="train.txt"):

    f = open(path, 'w')
    f.write('\n')
    for i in range(epochs):
        print("epoch",i)
        model = Model(alpha, beta, iteration, num_particles)
        model.run()
        s = str(model.gbest.pbest_dist)
        f.write(s)
        f.write('\n')

    f = open(path, 'r')
    count = 0
    s = f.readlines()
    for dist in s:
        if dist[0:7] == "155.153":
            count += 1
    print("accuracy ",(int)(100*(count / epochs)),'%',sep='')
    f.close()

# 得到城市数据
cities = read_cities("data/city.csv")
# 得到城市距离矩阵
MATRIX = get_MATRIX(cities.shape[0], cities)

alpha = 1.377429662907139
beta = 0.4538259013172141
iteration=300
num_particles=251
#1.5 0.2 300 200

model = Model(alpha, beta, iteration, num_particles)
model.run()
show_image(model.gbest)

# train(20,20)
#test(alpha,beta,iteration,num_particles,10)



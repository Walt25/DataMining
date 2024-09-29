# Import các thư viện cần thiết
from matplotlib.pylab import pinv
import numpy as np
import matplotlib.pyplot as plt
import seaborn; seaborn.set_style("whitegrid")
from sklearn.linear_model import LinearRegression

# Đọc dữ liệu từ file data.txt
data = np.loadtxt('data/data.txt', delimiter=',')
x, y = data[:, :2], data[:, 2]

# Xem trước 5 dữ liệu đầu tiên
print("Dữ liệu đầu vào (x):")
print(x[:5])
print("Giá nhà (y):")
print(y[:5])

# 1. Trực quan hóa dữ liệu
fig = plt.figure(figsize=(14, 5))
fig.subplots_adjust(wspace=0.3)

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# Scatter plots:
ax1.scatter(x[:, 0], y, marker='x', c='r', alpha=0.5, s=20)
ax1.set_xlabel('Sq.ft of house')
ax1.set_ylabel('House price')
ax1.set_xlim(0, 6000)

ax3 = ax1.twiny()
ax3.scatter(x[:, 1], y, marker='o', c='b', alpha=0.4, s=20)
ax3.set_xlabel('Number of bedrooms')
ax3.set_xlim(0, 6)

# Histograms:
ax2.hist(x[:, 0], alpha=0.4, edgecolor='b', linewidth=0.8)
ax2.set_xlabel('Sq.ft of house')
ax2.set_ylabel('Total count')
ax2.set_xlim(0, 6000)

ax4 = ax2.twiny()
ax4.hist(x[:, 1], alpha=0.4, color='r', edgecolor='r', linewidth=0.8)
ax4.set_xlabel('Number of bedrooms')
ax4.set_xlim(0, 6)

# 2. Chuẩn hóa dữ liệu
def featureNormalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

# In dữ liệu trước khi chuẩn hóa
print("Dữ liệu trước khi chuẩn hóa:")
print(x[:5])

# Chuẩn hóa dữ liệu
X, mean, sigma = featureNormalize(x)

# Dữ liệu sau khi chuẩn hóa
print("Dữ liệu sau khi chuẩn hóa:")
print(X[:5])

# Trực quan hóa dữ liệu sau khi chuẩn hóa
fig = plt.figure(figsize=(7, 5))
plt.hist(X[:, 0], alpha=0.4, edgecolor='b', linewidth=0.8, label='sq.ft')
plt.hist(X[:, 1], alpha=0.4, color='r', edgecolor='r', linewidth=0.8, label='bedrooms')
plt.xlabel('Features normalized')
plt.ylabel('Total count')
plt.xlim(-4, 4)
pst = plt.legend(loc='best', frameon=True)
pst.get_frame().set_edgecolor('k')

plt.show()

# 3. Gradient Descent
# Thêm cột 1 vào X
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Khởi tạo tham số
theta = np.zeros(X.shape[1])
iterations = 1500
alpha = 0.01

# Tính Cost Function J
def computeCost(X, y, theta):
    m = len(y)
    cost = (1/(2*m)) * np.sum(np.square(X.dot(theta) - y))
    return cost

cost = computeCost(X, y, theta)
print(f"Cost function: {cost}")

# Định nghĩa hàm Gradient Descent
def gradientDescent(X, y, theta, alpha, iterations):
    m = len(y)
    cost = np.zeros(iterations)
    thetaMod = theta.copy()
    
    for i in range(iterations):
        thetaMod = thetaMod - (alpha/m) * (X.T.dot(X.dot(thetaMod) - y))
        cost[i] = computeCost(X, y, thetaMod)
    
    return thetaMod, cost

# Chạy Gradient Descent
gradient, cost = gradientDescent(X, y, theta, alpha, iterations)
print('Giá trị theta tự tính bằng Gradient Descent:')
print(f"{'theta[0]:':<10} {gradient[0]:,.2f}")
print(f"{'theta[1]:':<10} {gradient[1]:,.2f}")
print(f"{'theta[2]:':<10} {gradient[2]:,.2f}")

# Vẽ hàm Loss
plt.plot(cost)
plt.title('Cost function over iterations')
plt.xlabel('Iterations')
plt.ylabel('Cost function J')
plt.show()

# 4. Nghiệm với Scikit-learn
reg = LinearRegression(fit_intercept=True)
model = reg.fit(x, y)
print('Giá trị theta tính bằng scikit-learn:')
print(f"{'theta[0]:':<10} {model.intercept_:,.2f}")
print(f"{'theta[1]:':<10} {model.coef_[0]:,.2f}")
print(f"{'theta[2]:':<10} {model.coef_[1]:,.2f}")

# 4.1. Tuỳ chọn Learning Rate cho Gradient Descent
learningRates = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
fig, ax = plt.subplots(figsize=(7, 5))

for alpha in learningRates:
    gradientNew, costNew = gradientDescent(X, y, theta, alpha, 50) 
    ax.plot(costNew, label='alpha = {0}'.format(alpha))

ax.set_ylabel(r"Cost function - $J(\theta)$")
ax.set_xlabel('Iterations')
pst = plt.legend(loc='best', frameon=True)
pst.get_frame().set_edgecolor('k');
plt.title('So sánh Cost Function với các Learning Rate khác nhau')
plt.show()

# 5. Thực hiện dự đoán
# Chạy lại thuật toán với LR tối ưu = 0.03
gradient, cost = gradientDescent(X, y, theta, 0.03, 500)  

# Normalizing parameters:
paramsNorm = (np.array([1650, 3]) - mean) / sigma

# Thêm cột 1 vào paramsNorm
params = np.hstack((np.array([1]), paramsNorm))

# Dự đoán giá nhà
predict_gd = np.dot(gradient, params)
print("\nDự đoán giá nhà:")
print(f"A 3 bedroom / 1650 sqft house will cost (GD) ${predict_gd:,.2f}")

# Nghiệm bằng Normal Equation
def normalEqn(X, y):
    theta = np.zeros((X.shape[1], 1))  # Khởi tạo theta
    
    # Tính toán giá trị theta bằng công thức Normal Equation
    theta = pinv(X.T.dot(X)).dot(X.T).dot(y)  # Sử dụng pinv cho ma trận nghịch đảo
    return theta

# Thêm cột 1 vào ma trận X
X = np.hstack((np.ones((len(x), 1)), x))

# Tính nghiệm Normal Equation
  # Thay đổi giá trị của lambda nếu cần
thetaN = normalEqn(X, y)
print("\nGiá trị theta tính bằng Normal Equation:")
print(f"{'theta[0]:':<10} {thetaN[0]:,.2f}")
print(f"{'theta[1]:':<10} {thetaN[1]:,.2f}")
print(f"{'theta[2]:':<10} {thetaN[2]:,.2f}")

# Dự đoán giá nhà bằng Normal Equation
params = np.array([1, 1650, 3])  
predict_ne = np.dot(thetaN, params)
print(f"A 3 bedroom / 1650 sqft house will cost (NE) ${predict_ne:,.2f}")

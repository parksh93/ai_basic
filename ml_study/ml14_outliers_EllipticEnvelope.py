import numpy as np
from sklearn.covariance import EllipticEnvelope #이상치 탐지

outliers_data = np.array([
    -50, -10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 50, 100
])

print(outliers_data.shape) # (21, )
outliers_data = outliers_data.reshape(-1, 1)
print(outliers_data.shape) # (21, 1)

# EllipticEnvelope
outliers = EllipticEnvelope(contamination=.3)
outliers.fit(outliers_data)
result = outliers.predict(outliers_data)

# 시각화
# import matplotlib.pyplot as plt
# plt.boxplot()
# plt.show()
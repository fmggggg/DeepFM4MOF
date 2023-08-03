'''
# Time   : 2023/08/02
# Author : Minggao Feng
'''
from model import DeepFM
from utils import create_hmof_dataset
import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers
from sklearn.metrics import r2_score,mean_absolute_error
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepFM4MOF')
    parser.add_argument('--coldstart', type=str, default='True')
    parser.add_argument('--property', type=int, default=1)
    parser.add_argument('--targetfrac', type=float, default=0.9)
    args = parser.parse_args()
    ifcoldstart =True if args.coldstart.lower()=="true" else False
    propertyID=args.property
    target_frac=args.targetfrac
    print('ifcoldstart=',ifcoldstart)
    if ifcoldstart:
        print('propertyID=',propertyID)
        print('targetfrac=',target_frac)
    file_path = 'dataset/28ads-8206hMOFs.csv'
    feature_columns, (X_train, y_train), (X_test, y_test),test_test_target = create_hmof_dataset(file_path,ifcoldstart,propertyID,target_frac, embed_dim=8,test_size=0.2)

    k = 10
    w_reg = 1e-4
    v_reg = 1e-4
    hidden_units = [256, 128, 64]
    output_dim = 1
    activation = 'relu'

    model = DeepFM(feature_columns, k, w_reg, v_reg, hidden_units, output_dim, activation)
    optimizer = optimizers.Adam(0.01)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)

    # train
    results = []
    y_train=tf.constant(y_train, dtype = tf.float32, shape=[len(y_train),1])
    y_test=tf.constant(y_test, dtype = tf.float32, shape=[len(y_test),1])
    for i in range(1000):
        with tf.GradientTape() as tape:
            y_pre = model(X_train)
            loss = tf.reduce_mean(tf.square(y_pre-y_train))
            total_error = tf.reduce_sum(tf.square(tf.subtract(y_train, tf.reduce_mean(y_train)))) 
            unexplained_error = tf.reduce_sum(tf.square(tf.subtract(y_train, y_pre))) 
            trainr2 = -tf.subtract(1, tf.truediv(unexplained_error, total_error))
            test_loss =tf.reduce_mean(tf.square(model(X_test)-y_test))
            test_mae =tf.reduce_mean(tf.abs(model(X_test)-y_test))
            pre = model(X_test)
            test_total_error = tf.reduce_sum(tf.square(tf.subtract(y_test, tf.reduce_mean(y_test)))) 
            test_unexplained_error = tf.reduce_sum(tf.square(tf.subtract(y_test, pre))) 
            test_r2 = tf.subtract(1, tf.truediv(test_unexplained_error, test_total_error)) 
            if (i+1)%10 == 0:    
                print('Iteration: No.{}'.format(i+1))
                print('train_MSE:{}'.format(loss.numpy()))
                print('train_R2:{}'.format(trainr2.numpy()))
                print('test_MSE:{}'.format(test_loss.numpy()))
                print('test_MAE:{}'.format(test_mae.numpy()))
                print('test_R2:{}'.format(test_r2.numpy()))

        grad = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grad, model.variables))
        results.append([i, test_loss.numpy(),test_r2.numpy(),test_mae.numpy()])

    #save the results
    pd.DataFrame(results, columns=['Iteration', 'test_mse', 'test_r2', 'test_mae'])\
        .to_csv('log/DeepFM_log.csv', index=False)
    
    #save the model
    tf.saved_model.save(model, "traind_models/traind_model")

    if not ifcoldstart:
        #fig
        print("-----------------------------generating figure----------------------------")
        X_test=tf.constant(X_test, dtype = tf.float32)
        pre_test=model.call(X_test)
        df_pre_test=pd.DataFrame(np.array(pre_test),columns=['pre'])
        df_y_test=pd.DataFrame(np.array(y_test),columns=['test'])
        test_total=df_pre_test.join(df_y_test)
        x=np.array(df_pre_test)
        y=np.array(df_y_test)
        x=x.reshape(1,-1)
        y=y.reshape(1,-1)
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)
        #plot
        plt.rc('font',family='Times New Roman') 
        fig, ax = plt.subplots()
        plt.scatter(y, x,c=z,  s=0.5,cmap='rainbow')
        plt.colorbar(label='Point Density')
        plt.title("imputation_test")
        plt.xlabel('test') 
        plt.ylabel('pred')  
        plt.xlim(0,1)
        plt.ylim(0,1)
        xpoints = np.array([0, 1])
        ypoints = np.array([0, 1])
        plt.plot(xpoints, ypoints,linewidth=1,linestyle="--",c="black")
        plt.savefig("log/imputation_test.png", format="png")
        plt.show()


    #cold-start test
    if ifcoldstart:
        print('-----------------------------Cold-start test--------------------------------')
        test_test_X = test_test_target.drop(['label'], axis=1).values
        test_test_y = test_test_target['label']
        test_test_X=tf.convert_to_tensor(test_test_X, tf.float32, name='inputs')
        pre_test=model.call(test_test_X)#predict the remaining test data of the target property
        df_pre_test=pd.DataFrame(np.array(pre_test),columns=['pre'])
        df_y_test=pd.DataFrame(np.array(test_test_y),columns=['test'])
        test_total=df_pre_test.join(df_y_test)
        print(f"coldstart-MAE：{mean_absolute_error(test_total['test'], test_total['pre'])}")
        # print(f"根均方误差(RMSE)：{np.sqrt(mean_squared_error(rf['actual'], rf['predicted']))}")
        print(f"coldstart-R^2：{r2_score(test_total['test'], test_total['pre'])}")

        # save the prediction results
        test_total.to_excel('log/test_cold-start.xlsx')

        #generate the fig
        # Calculate the point density
        print("-------------------------------generating figure-----------------------------")
        x=np.array(df_pre_test)
        y=np.array(df_y_test)
        x=x.reshape(1,-1)
        y=y.reshape(1,-1)
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)
        #plot
        plt.rc('font',family='Times New Roman') 
        fig, ax = plt.subplots()
        plt.scatter(y, x,c=z,  s=0.5,cmap='rainbow')
        plt.colorbar(label='Point Density')
        plt.title("cold-start_test")
        plt.xlabel('test') 
        plt.ylabel('pred')  
        plt.xlim(0,1)
        plt.ylim(0,1)
        xpoints = np.array([0, 1])
        ypoints = np.array([0, 1])
        plt.plot(xpoints, ypoints,linewidth=1,linestyle="--",c="black")
        plt.savefig("log/cold-start_test.png", format="png")
        plt.show()

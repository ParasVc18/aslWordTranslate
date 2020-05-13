import os
import sys
import argparse
from train_utils import make_label,load_data, load_data_testing, build_model

def main(dirname,dirname2):
    x_train,y_train,x_val,y_val=load_data(dirname)
    x_test,y_test=load_data_testing(dirname2)
    # print('\nTRAINING DATA X-')
    # print(x_train.shape)
    # print('\nTRAINING DATA Y-')
    # print(y_train.shape)
    # print('\nVALIDATION DATA X-')
    # print(x_val.shape)
    # print('\nVALIDATION DATA Y-')
    # print(y_val.shape)
    # print('\n')
    # print('\TEST DATA X-')
    # print(x_test.shape)
    # print('\nTEST DATA Y-')
    # print(y_test.shape)
    # print('\n')
    model=build_model(y_train.shape[1])
    print('Training stage')
    print('==============')
    model.fit(x_train,y_train,epochs=200,batch_size=16,validation_data=(x_val,y_val))
    score, acc = model.evaluate(x_test,y_test,batch_size=16,verbose=0)
    print('Test performance: accuracy={0}, loss={1}'.format(acc, score))
    model.save('model.h5')
    model.summary()
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Training Model')
    parser.add_argument("--input_train_path",help=" ")
    parser.add_argument("--input_test_path",help=" ")
    args=parser.parse_args()
    input_train_path=args.input_train_path
    input_test_path=args.input_test_path
    main(input_train_path,input_test_path)

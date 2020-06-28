


class DMTL(MyModel):
    def __init__(self,params, base_model, label, loss, second_model,data=self.prepare_data(2)):
        self.phase=phase
        labels = ['h','v','a']
        self.label_index = labels.index(label)
        default_params = {"agumentation": False, "scale": False, "dense_activation": "relu", "regularizition": 0.0
            , "dropout": 0.0, "optimizer": "adam", "number_of_dense": 1, "balancer": "None", "batch_size": 32,
                          "nn": 1024}
        default_params.update(params)
        Model = base_model
        params = default_params

        self.agumnetation = params["agumentation"]

        ############ Creating CNN ##############
        optimizer = params["optimizer"]
        for l in second_model.layers:
                l.trainable = False
        y=second_model.layers[-2].output
        out=[]
        for k in range(len(labels)):
            x=y
            for i in range(params["number_of_dense"][k]):
                if params["regularizition"][k] != 0.0:
                    x = Dense(params["nn"][k], activation=params["dense_activation"][k],trainable=(self.label_index==k),
                              kernel_regularizer=l2(params["regularizition"][k]), name="dense_" + labels[k] + str(i))(x)
                else:
                    x = Dense(params["nn"][k], activation=params["dense_activation"][k],trainable=(self.label_index==k),
                              name="dense_" + labels[k] + str(i))(x)
                if params["dropout"][k] != 0.0:
                    x = Dropout(params["dropout"][k],trainable=(self.label_index==k), name="dropout_" + labels[k] + str(i))(x)
            x = Dense(1, activation="sigmoid",trainable=(self.label_index==k),name="class_"+lables[k])(x)
            out.append(x)

        model = tf.keras.Model(second_model.input,out)
        model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

        self.__model = model
        self.__data = data
        self.balancer = params["balancer"]
        self.__number_of_dense = params["number_of_dense"]
        tmp = list(params.values())
        keys = list(params.keys())
        for i in range(len(keys)):
            if keys[i] in ["dense_activation", "regularizition", "dropout", "number_of_dense"]:
                tmp[i] = tmp[i][label_index]
            print(tmp[i])

        self.details = [keys[i] + ":" + str(tmp[i]) for i in range(len(keys))]


    def __init__(self,params, base_model ,loss,data=self.prepare_data(1)):
        self.name = name
        self.phase=phase
        self.label_index = label_index
        default_params = {"agumentation": False, "scale": False, "dense_activation": "relu", "regularizition": 0.0
            , "dropout": 0.0, "optimizer": "adam", "number_of_dense": 1, "balancer": "None", "batch_size": 32,
                          "nn": 1024}
        default_params.update(params)
        Model = base_model
        params = default_params
        d_a = load_data(label="a", phase="search")
        d_h = load_data(label="h", phase="search")
        d_v = load_data(label="v", phase="search")
        y = {"y_val", "y_train", "y_test"}
        x = {"x_val", "x_train", "x_test", "x_val_128","x_train_128"}

        regularization = not (params["regularizition"] == 0.0)
        dropout = not (params["dropout"] == 0.0)
        self.agumnetation = params["agumentation"]

        ############ Creating CNN ##############
        optimizer = params["optimizer"]
        inp = Input((64, 64, 1))
        con = concatenate([inp, inp, inp])

        model = Model(include_top=False, weights='imagenet', input_tensor=con)
        out = []

        y=Flatten()(model.layers[-1].ouput)
        for i in range(params["number_of_dense"]):
            if regularization:
                x = Dense(params["nn"], activation=params["dense_activation"],
                        kernel_regularizer=l2(params["regularizition"]), name="BigDense")(x)
            else:
                x = Dense(params["nn"], activation=params["dense_activation"],
                              name="BigDense")(x)
            if dropout:
                x = Dropout(params["dropout"], name="BigDrop")(x)
            x = Dense(1, activation="sigmoid",name="classification")(x)
            out.append(x)

        out=concatenate(array)
        model = tf.keras.Model(inp,array)
        model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

        self.__model = model
        self.__data =data
        self.__checkpoint=CallBacks.DMTL_ModelCheckpoint(self.__data["x_val"], self.__data["y_val"], self.__model, name_of_best_weight,self.label_index)
        self.balancer = params["balancer"]
        self.__number_of_dense = params["number_of_dense"]
        tmp = list(params.values())
        keys = list(params.keys())
        for i in range(len(keys)):
            if keys[i] in ["dense_activation", "regularizition", "dropout", "number_of_dense"]:
                tmp[i] = tmp[i][label_index]
            print(tmp[i])

        self.details = [keys[i] + ":" + str(tmp[i]) for i in range(len(keys))]
    def prepare_data(self,phase):
        d_a = load_data(label="a", phase="search")
        d_h = load_data(label="h", phase="search")
        d_v = load_data(label="v", phase="search")
        y = {"y_val", "y_train", "y_test"}
        x = {"x_val", "x_train", "x_test", "x_val_128", "x_train_128"}
        data = {}
        for t in y:
            if phase == 1:
                data[t] = np.array([[d_a[t][i][0], d_v[t][i][0], d_h[t][i][0]] for i in range(len(d_a[t]))])
            else:
                data[t] = [d_a[t], d_v[t], d_h[t]]
        for t in x:
            data[t] = d_a[t]
        self.batch_size = params["batch_size"]
        if params['agumentation']:
            data["x_val"] = ld.normalize(data["x_val"])
            data["x_test"] = ld.normalize(data["x_test"])
        elif params["scale"]:
            data["x_val"] = ld.normalize(data["x_val"])
            data["x_test"] = ld.normalize(data["x_test"])
            data["x_train"] = ld.normalize(data["x_train"])
        return data

    def evaluate(self):
        if self.phase == 1:
            tmp = model.get_model().predict(self.__data["x_test"])
            a = tmp[:, 0] > 0.5
            a = a * 1
            h = tmp[:, 2] > 0.5
            h = h * 1
            v = tmp[:, 1] > 0.5
            v = v * 1
            a = a == self.__data["y_test"][:, 0]
            h = h == self.__data["y_test"][:, 2]
            v = v == self.__data["y_test"][:, 1]
            a_acc = np.sum(a) / len(a)
            v_acc = np.sum(v) / len(v)
            h_acc = np.sum(h) / len(h)
            print(a_acc, v_acc, h_acc)
        else:
            ans = model.evaluate(self.__data["x_test"],self.__data["y_test"])
            print(ans)

    @staticmethod
    def k_fold_train(k,epochs,label, params, name, load_best_weigth, verbose, TensorB, name_of_best_weight,base_model):
        data_tmp = [None, None, None]
        flag = None
        if params['agumentation']:
            data_tmp[0] = ld.load_data('a', phase="aug_evaluation")
            data_tmp[1] = ld.load_data('v', phase="aug_evaluation")
            data_tmp[2] = ld.load_data('h', phase="aug_evaluation")
            flag = True
        else:
            data_tmp[0] = ld.load_data('a', phase="evaluation")
            data_tmp[1] = ld.load_data('v', phase="evaluation")
            data_tmp[2] = ld.load_data('h', phase="evaluation")
            flag = False
        data = {}
        data['x'] = data_tmp[0]['x']
        data['y'] = [data_tmp[0]['y'], data_tmp[1]['y'], data_tmp[2]['y']]
        size = len(data['x']) // k
        tmp_idx = np.arange(data['x'].shape[0])
        np.random.shuffle(tmp_idx)
        x = data['x'][tmp_idx]
        y = [data['y'][0][tmp_idx], data['y'][1][tmp_idx], data['y'][2][tmp_idx]]
        acc_vec = []
        np.save("x.npy",x)
        np.save("y.npy",y)
        for i in range(k):
            x_test = x[i * size:(i + 1) * size]

            y_test = [y[0][i * size:(i + 1) * size], y[1][i * size:(i + 1) * size], y[2][i * size:(i + 1) * size]]
            x_train = np.append(x[0:i * size], x[(i + 1) * size:], axis=0)
            y_train = [np.append(y[0][0:i * size], y[0][(i + 1) * size:], axis=0),
                       np.append(y[1][0:i * size], y[1][(i + 1) * size:], axis=0),
                       np.append(y[2][0:i * size], y[2][(i + 1) * size:], axis=0)]

            y_test1 = np.array([[y[0][i][0],y[1][i][0],y[2][i][0]] for i in range(i * size,(i + 1) * size)])
            y_train1=[[y[0][i][0], y[1][i][0], y[2][i][0]] for i in range(i * size)]+[[y[0][i][0], y[1][i][0], y[2][i][0]] for i in range((i + 1) * size,len( y[2]))]
            y_train1=np.array(y_train1)
            tmp = random.sample(range(len(x_train)), 232)
            x_val = []
            y_val = [[], [], []]
            y_val1=[]
            for j in tmp:
                x_val.append(x_train[j])
                y_val[0].append(y_train[0][j])
                y_val[1].append(y_train[1][j])
                y_val[2].append(y_train[2][j])
                y_val1.append([y_train[0][j][0],y_train[1][j][0],y_train[2][j][0]])
            y_val1=np.array(y_val1)
            x_val = np.array(x_val)
            x_train = np.delete(x_train, tmp, axis=0)
            y_train[0] = np.delete(y_train[0], tmp, axis=0)
            y_train[1] = np.delete(y_train[1], tmp, axis=0)
            y_train[2] = np.delete(y_train[2], tmp, axis=0)
            y_train1=np.delete(y_train1, tmp, axis=0)
            ##########fixing data#########
            data2 = ld.fix_data(flag, x_train, y_train, x_val, y_val, x_test, y_test)
            data1 = ld.fix_data(flag, x_train, y_train1, x_val, y_val1, x_test, y_test1)



            # epochs = 500
            par={"agumentation": False, "scale": False, "dense_activation":["sigmoid","",""] , "regularizition":[0.01,0.01,0.01] ,
                "dropout":[0.0,0.0,0.0] , "optimizer": "adadelta", "number_of_dense": [1,1,1], "balancer":"None",
                "batch_size":64,"nn":[4096*4,0,0]}


            model = MyModel(par, base_model, loss="mse", data=data1)

            model.train(200, False, verbose, TensorB, name_of_best_weight + str(i) + ".h5", "k_fold")
            model.get_model().save("m.h5")
            second_model = load_model("m.h5")
            model = MyModel(params, name, base_model, label_index, saver, loss="binary_crossentropy", second_model=second_model,phase=2,data=data2)

            model.train(epochs, load_best_weigth, verbose, TensorB, name_of_best_weight + str(i) + ".h5", "k_fold", None)
            ans = model.get_model().evaluate(data2["x_test"], data2["y_test"])
            acc = ans[4:]
            acc_vec.append(acc)
            print(acc)
            model.clear()
            del model
        return acc_vec



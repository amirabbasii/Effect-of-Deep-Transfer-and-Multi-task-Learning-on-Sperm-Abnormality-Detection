

class DTL(MyModel):
    def __init__(self, params,base_model):
        default_params = {"agumentation": False, "scale": False, "dense_activation": "relu", "regularizition": 0.0
            , "dropout": 0.0, "optimizer": "adam", "number_of_dense": 1, "balancer": "None", "batch_size": 32}
        default_params.update(params)
        Model = base_model
        params = default_params
        data = load_data(label=label, phase="search")
        self.batch_size = params["batch_size"]
        if params['agumentation']:
            data["x_val"] = ld.normalize(data["x_val"])
            data["x_test"] = ld.normalize(data["x_test"])
        elif params["scale"]:
            data["x_val"] = ld.normalize(data["x_val"])
            data["x_test"] = ld.normalize(data["x_test"])
            data["x_train"] = ld.normalize(data["x_train"])
        regularization = not (params["regularizition"] == 0.0)

        dropout = not (params["dropout"] == 0.0)

        self.agumnetation = params["agumentation"]

        ############ Creating CNN ##############
        optimizer = params["optimizer"]
        inp = Input((64, 64, 1))
        con = concatenate([inp, inp, inp])
        model = Model(include_top=False, weights='imagenet', input_tensor=con)
        x = Flatten()(model.layers[-1].output)

        for i in range(params["number_of_dense"]):
            if regularization:
                x = Dense(params["nn"], activation=params["dense_activation"],
                          kernel_regularizer=l2(params["regularizition"]))(x)
            else:
                x = Dense(params["nn"], activation=params["dense_activation"])(x)
            if dropout:
                x = Dropout(params["dropout"])(x)
        x = Dense(1, activation="sigmoid", name="classification")(x)
        model = tf.keras.Model(model.input, x)
        model.compile(optimizer=optimizer, metrics=["accuracy"], loss=params["loss"])

        self.__model = model
        self.__data = data
        self.__checkpoint=CallBacks.DTL_ModelCheckpoint(self.__data["x_val"], self.__data["y_val"], self.__model, name_of_best_weight)
        self.balancer = params["balancer"]
        self.__number_of_dense = params["number_of_dense"]
        self.details = [list(params.keys())[i] + ":" + str(list(params.values())[i]) for i in range(len(params))]

    def evaluate(self):
        X = self.__data["x_test"]
        y = self.__data["y_test"]
        y_pred1 = self.__model.predict(X)
        y_pred = y_pred1 > 0.5
        y_pred = y_pred * 1
        c_matrix = confusion_matrix(y, y_pred)
        precision = c_matrix[0, 0] / sum(c_matrix[0])
        recall = c_matrix[0, 0] / sum(c_matrix[:, 0])
        acc = np.sum(c_matrix.diagonal()) / np.sum(c_matrix)
        f_half = 1.25 * precision * recall / (.25 * precision + recall)
        g_mean = math.sqrt(precision * recall)
        TP = c_matrix[0][0]
        TN = c_matrix[1][1]
        FP = c_matrix[0][1]
        FN = c_matrix[1][0]
        mcc = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        auc = roc_auc_score(y, y_pred)
        return [acc,precision,recall,f_half,g_mean,auc,mcc]
    @staticmethod
    def k_fold(k, epochs, label, params, load_best_weigth, verbose, TensorB, name_of_best_weight,base_model):
        data = None
        flag = None
        results=[]
        if params['agumentation']:
            data = ld.load_data(label, phase="aug_evaluation")
            flag = True
        else:
            data = ld.load_data(label, phase="evaluation")
            flag = False
        size = len(data['x']) // k
        tmp_idx = np.arange(data['x'].shape[0])
        np.random.shuffle(tmp_idx)
        x = data['x'][tmp_idx]
        y = data['y'][tmp_idx]
        np.save("x.npy", x)
        np.save("y.npy", y)
        acc_vec = []
        for i in range(k):
            x_test = x[i * size:(i + 1) * size]
            y_test = y[i * size:(i + 1) * size]
            x_train = np.append(x[0:i * size], x[(i + 1) * size:], axis=0)
            y_train = np.append(y[0:i * size], y[(i + 1) * size:], axis=0)
            tmp = random.sample(range(len(x_train)), 232)
            x_val = []
            y_val = []
            for j in tmp:
                x_val.append(x_train[j])
                y_val.append(y_train[j])
            x_val = np.array(x_val)
            y_val = np.array(y_val)
            x_train = np.delete(x_train, tmp, axis=0)
            y_train = np.delete(y_train, tmp, axis=0)

            ##########fixing data#########
            data = ld.fix_data(flag, x_train, y_train, x_val, y_val, x_test, y_test)

            model = MyModel(params, data, base_model=base_model)
            model.train(epochs, load_best_weigth, verbose, TensorB, name_of_best_weight + str(i) + ".h5", "k_fold")
            results.append(model.evaluate())
            print(results[-1])
            model.clear()
            del model
        return results






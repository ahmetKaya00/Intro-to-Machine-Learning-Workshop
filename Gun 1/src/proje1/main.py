from prepocessing import handle_missing_values, encode_categorial_data,scale_data,split_data

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    data = pd.DataFrame({
        "Yaş": [25,45,30,35,40],
        "Cinsiyet": ["Kadın","Erkek","Erkek","Kadın",None],
        "Maaş": [3000,7000,None,5000,6000],
        "Deneyim": [2,20,5,10,15],
        "Departman": ["IT","Yönetim","Muhasebe","IT","Yönetim"],
        "Terfi":[0,1,0,1,1]
    })
    print("Orijinal Veri:")
    print(data)

    data = handle_missing_values(data)
    data = encode_categorial_data(data)

    scaler = StandardScaler()
    data[["Yaş","Maaş","Deneyim"]] = scaler.fit_transform(data[["Yaş","Maaş","Deneyim"]])

    x = data.drop("Terfi",axis=1)
    y = data["Terfi"]
    X_train,X_test,y_train,y_test = split_data(data)

    model = LogisticRegression()
    model.fit(X_train,y_train)

    new_employee = pd.DataFrame({
        "Yaş": [22],
        "Cinsiyet": [2],
        "Maaş": [7500],
        "Deneyim": [9],
        "Departman_Muhasebe": [0],
        "Departman_Yönetim": [1]
    })

    new_employee[["Yaş","Maaş","Deneyim"]] = scaler.transform(new_employee[["Yaş","Maaş","Deneyim"]])
    prediction = model.predict(new_employee)

    if prediction[0] == 1:
        print("Bu çalışan terfi alabilir!")
    else:
        print("Bu çalışan terfi alamaz!")



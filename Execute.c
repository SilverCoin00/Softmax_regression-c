#include "D:\Data\code_doc\AI_model_building\Softmax_regression\Core.h"

int main() {
    char file[] = "D:\\Data\\archive1\\flag.csv";
    Data_Frame* df = read_csv(file, 1000, ",");
    Dataset_2* ds = trans_dframe_to_dset2(df, "t");
    Standard_scaler* scaler = (Standard_scaler*)malloc(sizeof(Standard_scaler));
    scaler_fit(ds->x, NULL, ds->samples, ds->features, scaler, "Standard_scaler");
    scaler_transform(ds->x, NULL, ds->samples, ds->features, scaler, "Standard_scaler");
    //print_dataset2(ds, 4, 10, 15);
    Softmax_regression* model = (Softmax_regression*)malloc(sizeof(Softmax_regression));
    model->weights = init_weights(ds->features, ds->y_types, 1);
    model->data = ds;
    train(model, "NAG", 130, 4e-4, 32);
    free_sm_model(model);
    free_scaler(scaler, "Standard_scaler");
    free_data_frame(df);
    free_dataset2(ds);
    return 0;
}
import Apriori as apriori_module
import TimeSeries as ts_module   
import Clustering as clust_module
def main():
    #apriori_module.run_apriori()
    #apriori_module.interactive_predict_for_customer()
    #ts_module.time_series_forecast()
    clust_module.customer_segmentation()


if __name__ == "__main__":
    main()
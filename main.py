import Apriori as apriori_module
import TimeSeries as ts_module   
import Clustering as clust_module
def main():
    apriori_module.run_apriori()
    ts_module.time_series_forecast()
    clust_module.customer_segmentation()


if __name__ == "__main__":
    main()
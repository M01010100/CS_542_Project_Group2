import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from dataHelper import sort_member_numbers, load_data

def customer_segmentation():
    data = load_data()
    items_dict = sort_member_numbers(data)
    
    # Create feature matrix
    features = []
    member_ids = []
    
    for member, items in items_dict.items():
        member_ids.append(member)
        features.append({
            'total_items': len(items),
            'unique_items': len(set(items)),
            'avg_basket_size': len(items) / 1,  # Per transaction
            'diversity_ratio': len(set(items)) / len(items) if len(items) > 0 else 0
        })
    
    df = pd.DataFrame(features)
    df['member_id'] = member_ids
    
    # Standardize features
    scaler = StandardScaler()
    feature_cols = ['total_items', 'unique_items', 'avg_basket_size', 'diversity_ratio']
    scaled_features = scaler.fit_transform(df[feature_cols])
    
    # Determine optimal number of clusters
    optimal_k = find_optimal_clusters(scaled_features)
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_features)
    
    df['cluster'] = clusters
    
    # Display results
    display_cluster_summary(df, feature_cols)
    display_cluster_profiles(df, items_dict, feature_cols)
    visualize_clusters(scaled_features, clusters, df, feature_cols)
    display_cluster_recommendations(df, items_dict)
    
    return clusters, kmeans, df

def find_optimal_clusters(scaled_features, max_k=10):
    print("\n" + "="*80)
    print("FINDING OPTIMAL NUMBER OF CLUSTERS")
    print("="*80)
    
    inertias = []
    silhouette_scores = []
    K_range = range(2, min(max_k + 1, len(scaled_features)))
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(scaled_features, kmeans.labels_))
    
    # Find elbow using rate of change
    if len(inertias) > 2:
        rates = np.diff(inertias)
        optimal_k = np.argmax(rates) + 2  # +2 because we start at k=2
    else:
        optimal_k = 3
    
    print(f"\n Optimal number of clusters: {optimal_k}")
    print(f"   Silhouette Score: {silhouette_scores[optimal_k-2]:.3f}")
    
    # Plot elbow curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(K_range, inertias, 'bo-')
    ax1.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(K_range, silhouette_scores, 'go-')
    ax2.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return optimal_k

def display_cluster_summary(df, feature_cols):
    print("\n" + "="*80)
    print("CLUSTER SUMMARY STATISTICS")
    print("="*80)
    
    cluster_stats = df.groupby('cluster').agg({
        'total_items': ['count', 'mean', 'std'],
        'unique_items': ['mean', 'std'],
        'avg_basket_size': ['mean', 'std'],
        'diversity_ratio': ['mean', 'std']
    }).round(2)
    
    print("\n", cluster_stats)
    
    # Cluster size distribution
    cluster_sizes = df['cluster'].value_counts().sort_index()
    print("\n Cluster Sizes:")
    print("-" * 80)
    for cluster, size in cluster_sizes.items():
        percentage = (size / len(df)) * 100
        print(f"Cluster {cluster}: {size} customers ({percentage:.1f}%)")

def display_cluster_profiles(df, items_dict, feature_cols):
    print("\n" + "="*80)
    print("CUSTOMER SEGMENT PROFILES")
    print("="*80)
    
    segment_names = {
        0: "🛒 Light Shoppers",
        1: "💰 Regular Customers", 
        2: "⭐ Premium Shoppers",
        3: "🎯 Bulk Buyers"
    }
    
    for cluster in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster]
        
        print(f"\n{'='*80}")
        print(f"CLUSTER {cluster}: {segment_names.get(cluster, f'Segment {cluster}')}")
        print('='*80)
        
        # Statistics
        print(f"\n📊 Size: {len(cluster_data)} customers ({len(cluster_data)/len(df)*100:.1f}%)")
        print(f"\n📈 Average Metrics:")
        for col in feature_cols:
            avg_val = cluster_data[col].mean()
            print(f"   {col}: {avg_val:.2f}")
        
        # Top items in this cluster
        cluster_members = cluster_data['member_id'].tolist()
        all_items = []
        for member in cluster_members:
            if member in items_dict:
                all_items.extend(items_dict[member])
        
        if all_items:
            item_counts = pd.Series(all_items).value_counts().head(10)
            print(f"\n🔝 Top 10 Items:")
            for item, count in item_counts.items():
                percentage = (count / len(all_items)) * 100
                print(f"   {item}: {count} ({percentage:.1f}%)")

def visualize_clusters(scaled_features, clusters, df, feature_cols):
    print("\n Generating cluster visualizations...")

    # PCA for 2D visualization
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(scaled_features)
    
    # Create subplots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. PCA Scatter Plot
    ax1 = plt.subplot(2, 3, 1)
    scatter = ax1.scatter(features_2d[:, 0], features_2d[:, 1], 
                          c=clusters, cmap='viridis', alpha=0.6, s=50)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax1.set_title('Customer Segments (PCA)')
    plt.colorbar(scatter, ax=ax1, label='Cluster')
    
    # 2. Cluster sizes
    ax2 = plt.subplot(2, 3, 2)
    cluster_counts = df['cluster'].value_counts().sort_index()
    ax2.bar(cluster_counts.index, cluster_counts.values, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Number of Customers')
    ax2.set_title('Cluster Size Distribution')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Feature comparison across clusters
    ax3 = plt.subplot(2, 3, 3)
    cluster_means = df.groupby('cluster')[feature_cols].mean()
    cluster_means_normalized = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
    cluster_means_normalized.T.plot(kind='bar', ax=ax3)
    ax3.set_title('Normalized Feature Comparison')
    ax3.set_ylabel('Normalized Value')
    ax3.legend(title='Cluster', bbox_to_anchor=(1.05, 1))
    ax3.grid(axis='y', alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. Box plot for total items
    ax4 = plt.subplot(2, 3, 4)
    df.boxplot(column='total_items', by='cluster', ax=ax4)
    ax4.set_xlabel('Cluster')
    ax4.set_ylabel('Total Items')
    ax4.set_title('Total Items Distribution by Cluster')
    plt.suptitle('')  # Remove default title
    
    # 5. Box plot for unique items
    ax5 = plt.subplot(2, 3, 5)
    df.boxplot(column='unique_items', by='cluster', ax=ax5)
    ax5.set_xlabel('Cluster')
    ax5.set_ylabel('Unique Items')
    ax5.set_title('Unique Items Distribution by Cluster')
    plt.suptitle('')
    
    # 6. Heatmap of cluster characteristics
    ax6 = plt.subplot(2, 3, 6)
    cluster_means = df.groupby('cluster')[feature_cols].mean()
    sns.heatmap(cluster_means.T, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax6)
    ax6.set_title('Cluster Feature Heatmap')
    ax6.set_xlabel('Cluster')
    
    plt.tight_layout()
    plt.show()

def display_cluster_recommendations(df, items_dict):
    print("\n" + "="*80)
    print(" BUSINESS RECOMMENDATIONS BY SEGMENT")
    print("="*80)
    
    recommendations = {
        0: [
            " Target with entry-level promotions",
            " Send welcome emails with popular items",
            " Offer small basket discounts to increase purchase frequency"
        ],
        1: [
            " Implement loyalty program",
            " Suggest bundled products",
            " Send personalized shopping reminders"
        ],
        2: [
            " Offer premium product lines",
            " Provide VIP customer benefits",
            " Early access to new products"
        ],
        3: [
            " Bulk purchase discounts",
            " Free delivery for large orders",
            " Volume-based rewards program"
        ]
    }
    
    for cluster in sorted(df['cluster'].unique()):
        cluster_size = len(df[df['cluster'] == cluster])
        avg_items = df[df['cluster'] == cluster]['total_items'].mean()
        
        print(f"\n{'='*80}")
        print(f"CLUSTER {cluster} ({cluster_size} customers, avg {avg_items:.0f} items)")
        print('='*80)
        
        if cluster in recommendations:
            print("\nRecommended Actions:")
            for rec in recommendations[cluster]:
                print(f"  • {rec}")

def compare_clustering_algorithms(scaled_features, df):
    print("\n" + "="*80)
    print("COMPARING CLUSTERING ALGORITHMS")
    print("="*80)
    
    # K-Means
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans_labels = kmeans.fit_predict(scaled_features)
    
    # DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(scaled_features)

    print("\n Algorithm Comparison:")
    print("-" * 80)
    print(f"K-Means:")
    print(f"  Clusters found: {len(set(kmeans_labels))}")
    print(f"  Silhouette Score: {silhouette_score(scaled_features, kmeans_labels):.3f}")
    
    if len(set(dbscan_labels)) > 1:
        print(f"\nDBSCAN:")
        print(f"  Clusters found: {len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)}")
        print(f"  Noise points: {list(dbscan_labels).count(-1)}")
        non_noise = dbscan_labels[dbscan_labels != -1]
        if len(set(non_noise)) > 1:
            print(f"  Silhouette Score: {silhouette_score(scaled_features[dbscan_labels != -1], non_noise):.3f}")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

if __name__ == "__main__":
    df = pd.read_csv('data/processed_data/cleaned_data.csv')
    os.makedirs('visuals', exist_ok=True)
    sns.set_style("whitegrid")

    # 1. Distribution of Price_in_Lakhs
    plt.figure(figsize=(10,6))
    sns.histplot(df['Price_in_Lakhs'], bins=30, kde=True)
    plt.title('Distribution of Price_in_Lakhs')
    plt.xlabel('Price_in_Lakhs')
    plt.ylabel('Frequency')
    plt.savefig('visuals/price_distribution.png', bbox_inches='tight')
    plt.show()
    plt.close()

    # 2. Size vs Price relationship + outliers
    plt.figure(figsize=(10,6))
    plt.subplot(1,2,1)
    sns.boxplot(y='Price_in_Lakhs', data=df)
    plt.title('Boxplot of Price_in_Lakhs (Outliers)')
    plt.subplot(1,2,2)
    sns.scatterplot(x='Size_in_SqFt', y='Price_in_Lakhs', data=df, alpha=0.1)
    plt.title('Size vs Price_in_Lakhs')
    plt.tight_layout()
    plt.savefig('visuals/size_price_relationship.png', bbox_inches='tight')
    plt.show()
    plt.close()

    # 3. Price/SqFt by BHK + Furnished Status
    plt.figure(figsize=(12,6))
    sns.barplot(x='BHK', y='Price_per_SqFt', hue='Furnished_Status', data=df)
    plt.title('Average Price per SqFt by BHK and Furnished Status')
    plt.xlabel('BHK')
    plt.ylabel('Average Price per SqFt (Scaled)')
    plt.legend(title='Furnished Status')
    plt.savefig('visuals/price_per_sqft_by_bhk_furnished.png', bbox_inches='tight')
    plt.show()
    plt.close()

    # 4. Avg Price by State and City
    state_avg = df.groupby('State')['Price_in_Lakhs'].mean().sort_values(ascending=False).head(10)
    city_avg = df.groupby('City')['Price_in_Lakhs'].mean().sort_values(ascending=False).head(10)
    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    sns.barplot(x=state_avg.values, y=state_avg.index)
    plt.title('Avg Price by State (Top 10)')
    plt.xlabel('Average Price_in_Lakhs')
    plt.ylabel('State (Encoded)')
    plt.subplot(1,2,2)
    sns.barplot(x=city_avg.values, y=city_avg.index)
    plt.title('Avg Price by City (Top 10)')
    plt.xlabel('Average Price_in_Lakhs')
    plt.ylabel('City (Encoded)')
    plt.tight_layout()
    plt.savefig('visuals/avg_price_by_state_city.png', bbox_inches='tight')
    plt.show()
    plt.close()

    # 5. Correlation Heatmap
    cols = [c for c in df.columns if not any(c.startswith(p) for p in ['Property_Type_','Facing_','Owner_Type_'])]
    plt.figure(figsize=(16,12))
    sns.heatmap(df[cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Numerical Features')
    plt.savefig('visuals/correlation_heatmap.png', bbox_inches='tight')
    plt.show()
    plt.close()

    # 6. Nearby Schools & Hospitals vs Price (boxplot - better than scatter)
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    sns.boxplot(x='Nearby_Schools', y='Price_in_Lakhs', data=df)
    plt.title('Nearby Schools vs Price')
    plt.xlabel('Number of Nearby Schools')
    plt.ylabel('Price_in_Lakhs')
    plt.subplot(1,2,2)
    sns.boxplot(x='Nearby_Hospitals', y='Price_in_Lakhs', data=df)
    plt.title('Nearby Hospitals vs Price')
    plt.xlabel('Number of Nearby Hospitals')
    plt.ylabel('Price_in_Lakhs')
    plt.tight_layout()
    plt.savefig('visuals/nearby_facilities_vs_price.png', bbox_inches='tight')
    plt.show()
    plt.close()

    # 7. Availability Status counts
    plt.figure(figsize=(8,6))
    sns.countplot(x='Availability_Status', data=df)
    plt.title('Count of Availability Status')
    plt.xlabel('Availability Status (Encoded)')
    plt.ylabel('Count')
    plt.savefig('visuals/availability_status_counts.png', bbox_inches='tight')
    plt.show()
    plt.close()

    # 8. Parking Space vs Price
    plt.figure(figsize=(10,6))
    sns.boxplot(x='Parking_Space', y='Price_in_Lakhs', data=df)
    sns.stripplot(x='Parking_Space', y='Price_in_Lakhs', data=df, alpha=0.05, size=1, color='black')
    plt.title('Parking Space vs Price_in_Lakhs')
    plt.xlabel('Parking Space')
    plt.ylabel('Price_in_Lakhs')
    plt.savefig('visuals/parking_space_vs_price.png', bbox_inches='tight')
    plt.show()
    plt.close()

    
    # 9. Good Investment distribution + Public Transport vs Price
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    sns.countplot(x='Good_Investment', data=df)
    plt.title('Good Investment Distribution')
    plt.xlabel('Good Investment (0=No, 1=Yes)')
    plt.ylabel('Count')
    plt.subplot(1,2,2)
    sns.boxplot(x='Public_Transport_Accessibility', y='Price_in_Lakhs', data=df)
    plt.title('Public Transport Accessibility vs Price')
    plt.xlabel('Public Transport Score (0=Low, 1=Med, 2=High)')
    plt.ylabel('Price_in_Lakhs')
    plt.tight_layout()
    plt.savefig('visuals/good_investment_public_transport.png', bbox_inches='tight')
    plt.show()
    plt.close()

    # 10. BHK vs Average Price
    bhk_avg = df.groupby('BHK')['Price_in_Lakhs'].mean().sort_index()
    plt.figure(figsize=(10,6))
    sns.barplot(x=bhk_avg.index, y=bhk_avg.values)
    plt.title('Average Price by BHK')
    plt.xlabel('BHK')
    plt.ylabel('Average Price_in_Lakhs')
    plt.savefig('visuals/bhk_vs_price.png', bbox_inches='tight')
    plt.show()
    plt.close()

    print("All visuals saved to visuals/ folder!")
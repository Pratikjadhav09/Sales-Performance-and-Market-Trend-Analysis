import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

df = pd.read_csv("C:/Users/Loq/OneDrive/Documents/tasks 6/sales_dataset.csv")

for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

df = df.drop_duplicates()

df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')

df['Profit_Calc'] = df['Total_Sales'] - (df['Quantity'] * df['Unit_Cost'])

top_products_revenue = (
    df.groupby('Product')['Total_Sales']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

top_products_quantity = (
    df.groupby('Product')['Quantity']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

low_products_revenue = (
    df.groupby('Product')['Total_Sales']
    .sum()
    .sort_values(ascending=True)
    .head(10)
)

df['Month'] = df['Order_Date'].dt.to_period('M')
df['Quarter'] = df['Order_Date'].dt.to_period('Q')

monthly_sales = df.groupby('Month')['Total_Sales'].sum()
quarterly_sales = df.groupby('Quarter')['Total_Sales'].sum()

channel_sales = df.groupby('Sales_Channel')['Total_Sales'].sum()

region_sales = df.groupby('Region')['Total_Sales'].sum()

highest_region = region_sales.idxmax(), region_sales.max()
lowest_region = region_sales.idxmin(), region_sales.min()

print("\nTop 10 Products by Revenue:\n", top_products_revenue)
print("\nTop 10 Products by Quantity:\n", top_products_quantity)
print("\nLowest Performing Products by Revenue:\n", low_products_revenue)
print("\nMonthly Sales:\n", monthly_sales)
print("\nQuarterly Sales:\n", quarterly_sales)
print("\nSales by Channel:\n", channel_sales)
print("\nRegional Sales:\n", region_sales)
print(f"\nHighest Revenue Region: {highest_region}")
print(f"Lowest Revenue Region: {lowest_region}")

top_products_revenue.plot(kind='bar', title="Top 10 Products by Revenue", figsize=(8,5))
plt.ylabel("Revenue")
plt.show()

monthly_sales.plot(kind='line', marker='o', title="Monthly Sales Trend", figsize=(8,5))
plt.ylabel("Revenue")
plt.show()

region_sales.plot(kind='bar', title="Sales by Region", figsize=(8,5))
plt.ylabel("Revenue")
plt.show()

channel_sales.plot(kind='pie', autopct='%1.1f%%', title="Sales by Channel", figsize=(6,6))
plt.ylabel("")
plt.show()

customer_orders = df.groupby('Customer_Name')['Order_ID'].nunique()

customer_spending = df.groupby('Customer_Name')['Total_Sales'].sum()

customer_segments = customer_orders.apply(lambda x: "Repeat" if x > 1 else "One-time")

avg_order_value = df['Total_Sales'].mean()

category_counts = df['Category'].value_counts()

print("\nCustomer Segmentation (Repeat vs One-time):\n", customer_segments.value_counts())
print("\nAverage Order Value:", avg_order_value)
print("\nMost Common Purchase Categories:\n", category_counts)

monthly_sales_indexed = monthly_sales.reset_index()
monthly_sales_indexed['Month'] = monthly_sales_indexed['Month'].astype(str)

monthly_sales_indexed['Month_Num'] = np.arange(len(monthly_sales_indexed))

X = monthly_sales_indexed[['Month_Num']]
y = monthly_sales_indexed['Total_Sales']
model = LinearRegression()
model.fit(X, y)

future_months = pd.DataFrame({'Month_Num': np.arange(len(monthly_sales_indexed), len(monthly_sales_indexed)+3)})
future_predictions = model.predict(future_months)

predicted_sales = pd.DataFrame({
    "Future_Month": ["Next_M1", "Next_M2", "Next_M3"],
    "Predicted_Sales": future_predictions
})

print("\nPredicted Sales for Next Quarter:\n", predicted_sales)

with pd.ExcelWriter("sales_analysis_results.xlsx") as writer:
    
    top_products_revenue.reset_index().to_excel(writer, sheet_name="Top Products Revenue", index=False)
    top_products_quantity.reset_index().to_excel(writer, sheet_name="Top Products Quantity", index=False)
    low_products_revenue.reset_index().to_excel(writer, sheet_name="Lowest Products Revenue", index=False)
    monthly_sales.reset_index().to_excel(writer, sheet_name="Monthly Sales", index=False)
    quarterly_sales.reset_index().to_excel(writer, sheet_name="Quarterly Sales", index=False)
    channel_sales.reset_index().to_excel(writer, sheet_name="Channel Sales", index=False)
    region_sales.reset_index().to_excel(writer, sheet_name="Regional Sales", index=False)

    pd.DataFrame([{"Region": highest_region[0], "Total_Sales": highest_region[1]}]).to_excel(
        writer, sheet_name="Highest Revenue Region", index=False
    )
    pd.DataFrame([{"Region": lowest_region[0], "Total_Sales": lowest_region[1]}]).to_excel(
        writer, sheet_name="Lowest Revenue Region", index=False
    )

    customer_orders.reset_index().rename(columns={"Order_ID":"Purchase_Frequency"}).to_excel(
        writer, sheet_name="Customer Frequency", index=False
    )
    customer_spending.reset_index().rename(columns={"Total_Sales":"Total_Spending"}).to_excel(
        writer, sheet_name="Customer Spending", index=False
    )
    customer_segments.reset_index().rename(columns={"Customer_Name":"Customer", 0:"Segment"}).to_excel(
        writer, sheet_name="Customer Segments", index=False
    )
    category_counts.reset_index().rename(columns={"index":"Category","Category":"Count"}).to_excel(
        writer, sheet_name="Purchase Categories", index=False
    )

    predicted_sales.to_excel(writer, sheet_name="Sales Forecast", index=False)

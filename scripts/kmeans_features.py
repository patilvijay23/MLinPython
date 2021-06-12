# Script for k-means feature dataset creation
# takes 4 csvs as input, this can be updated as per input data format and location

import pandas as pd

# Read all input datasets

df_sales = pd.read_csv('./../data/clustering_sales.csv')
df_customer = pd.read_csv('./../data/clustering_customer.csv')
df_product = pd.read_csv('./../data/clustering_product.csv')
df_payment = pd.read_csv('./../data/clustering_payment.csv')

# Overall level features

# merge datasets
df_sales = df_sales.merge(df_product[['product_id','category']], on=['product_id'])
df_sales = df_sales.merge(df_payment, on=['payment_type_id'])

# aggregate
df_features_overall = df_sales.groupby(['customer_id']).agg({
    'dollars':'sum',
    'qty':'sum',
    'order_id':'nunique',
    'product_id':'nunique',
    'payment_type_id':'nunique',
    'category':'nunique'
    })

# derived features
df_features_overall['aov'] = df_features_overall['dollars']/df_features_overall['order_id']
df_features_overall['aur'] = df_features_overall['dollars']/df_features_overall['qty']
df_features_overall['upt'] = df_features_overall['qty']/df_features_overall['order_id']

# rename columns
df_features_overall.columns = [
    'sales','units','orders','unique_products_bought','unique_payments_used',
    'unique_categories_bought','aov','aur','upt']

# Category level features

# Sales
# aggregate
df_category_features_s = df_sales.groupby(['customer_id','category']).agg({'dollars':'sum'}).reset_index()

# add overall sales
df_category_features_s = df_category_features_s.merge(df_features_overall[['sales']], on=['customer_id'])

# convert to %
df_category_features_s['sales_perc'] = df_category_features_s['dollars']/df_category_features_s['sales']

# pivot
df_category_features_s = df_category_features_s.pivot(index='customer_id', columns='category', values='sales_perc')

# rename columns
df_category_features_s.columns = [
    'category_a_sales','category_b_sales','category_c_sales','category_d_sales','category_e_sales']

# Units
# aggregate
df_category_features_u = df_sales.groupby(['customer_id','category']).agg({'qty':'sum'}).reset_index()

# add overall sales
df_category_features_u = df_category_features_u.merge(df_features_overall[['units']], on=['customer_id'])

# convert to %
df_category_features_u['units_perc'] = df_category_features_u['qty']/df_category_features_u['units']

# pivot
df_category_features_u = df_category_features_u.pivot(index='customer_id', columns='category', values='units_perc')

# rename columns
df_category_features_u.columns = [
    'category_a_units','category_b_units','category_c_units','category_d_units','category_e_units']

# Tender type level
# aggregate
df_payment_features = df_sales.groupby(['customer_id','payment_type']).agg({'dollars':'sum'}).reset_index()

# add overall sales
df_payment_features = df_payment_features.merge(df_features_overall[['sales']], on=['customer_id'])

# convert to %
df_payment_features['sales_perc'] = df_payment_features['dollars']/df_payment_features['sales']

# pivot
df_payment_features = df_payment_features.pivot(index='customer_id', columns='payment_type', values='sales_perc')

# rename columns
df_payment_features.columns = [
    'payment_cash','payment_credit','payment_debit','payment_gc','payment_others']

# Final Features dataset
# merge all
df_features = df_features_overall.merge(df_category_features_s,on='customer_id',how='left')

df_features = df_features.merge(df_category_features_u,on='customer_id',how='left')

df_features = df_features.merge(df_payment_features,on='customer_id',how='left')

df_features = df_features.merge(
    df_customer[['customer_id','email_subscribed','omni_shopper']].set_index('customer_id'),
    on='customer_id',how='left')

# Write output
df_features.to_csv('./../data/clustering_features.csv', index=True)
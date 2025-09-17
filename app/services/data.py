import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import os
from app.utils.logger import logger


class DataService:
    """Service for loading and processing customer and order data"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.orders_df = None
        self.customers_df = None
        self._load_data()

    def _load_data(self):
        """Load CSV data with error handling"""
        try:
            orders_path = os.path.join(self.data_dir, "orders.csv")
            customers_path = os.path.join(self.data_dir, "customers.csv")

            if not os.path.exists(orders_path) or not os.path.exists(customers_path):
                logger.warning("CSV files not found, using sample data")
                self._create_sample_data()
                return

            self.orders_df = pd.read_csv(orders_path)
            self.customers_df = pd.read_csv(customers_path)

            # Convert date columns
            self.orders_df['order_date'] = pd.to_datetime(self.orders_df['order_date'])
            self.orders_df['total'] = self.orders_df['qty'] * self.orders_df['price']

            logger.info(
                "Data loaded successfully",
                orders_count=len(self.orders_df),
                customers_count=len(self.customers_df)
            )

        except Exception as e:
            logger.error("Failed to load data", error=str(e))
            self._create_sample_data()

    def _create_sample_data(self):
        """Create sample data if CSV files are not available"""
        # Sample orders data
        orders_data = [
            ["C001", "SO-101", "2025-08-20", "CAKE-CHOC", 3, 12.50],
            ["C001", "SO-122", "2025-09-05", "COOK-OAT", 5, 2.10],
            ["C002", "SO-130", "2025-09-01", "JUICE-ORG", 10, 1.20],
            ["C003", "SO-140", "2025-07-30", "CAKE-CHOC", 1, 12.50],
            ["C003", "SO-155", "2025-09-10", "COFF-BEAN", 2, 7.90],
            ["C001", "SO-160", "2025-09-12", "CAKE-CHOC", 1, 12.50],
            ["C004", "SO-170", "2025-08-01", "TEA-GREEN", 4, 3.50],
        ]

        self.orders_df = pd.DataFrame(
            orders_data,
            columns=["customer_id", "order_id", "order_date", "sku", "qty", "price"]
        )
        self.orders_df['order_date'] = pd.to_datetime(self.orders_df['order_date'])
        self.orders_df['total'] = self.orders_df['qty'] * self.orders_df['price']

        # Sample customers data
        customers_data = [
            ["C001", "Gourmet Gateway", "HO.RE.CA", "West", "NET15"],
            ["C002", "Snack Shack", "Retail", "East", "PREPAID"],
            ["C003", "Daily Delights", "Retail", "North", "NET30"],
            ["C004", "Leaf & Cup", "Cafe", "South", "NET15"],
        ]

        self.customers_df = pd.DataFrame(
            customers_data,
            columns=["customer_id", "name", "segment", "territory", "credit_terms"]
        )

        logger.info("Sample data created")

    def get_customer_data(self, customer_id: str) -> Optional[Dict]:
        """Get comprehensive customer data"""
        if self.customers_df is None or self.orders_df is None:
            return None

        # Check if customer exists
        customer_row = self.customers_df[self.customers_df['customer_id'] == customer_id]
        if customer_row.empty:
            return None

        # Get customer info
        customer_info = customer_row.iloc[0].to_dict()

        # Get order history
        customer_orders = self.orders_df[self.orders_df['customer_id'] == customer_id]

        if customer_orders.empty:
            order_history = []
            order_summary = {
                "total_orders": 0,
                "total_spent": 0.0,
                "avg_order_value": 0.0,
                "last_order_date": None,
                "days_since_last_order": None
            }
        else:
            order_history = customer_orders.to_dict('records')

            # Calculate summary metrics
            total_spent = customer_orders['total'].sum()
            last_order_date = customer_orders['order_date'].max()
            days_since_last_order = (datetime.now() - last_order_date).days

            order_summary = {
                "total_orders": len(customer_orders),
                "total_spent": float(total_spent),
                "avg_order_value": float(total_spent / len(customer_orders)),
                "last_order_date": last_order_date.strftime('%Y-%m-%d'),  # Convert to string
                "days_since_last_order": days_since_last_order
            }

        return {
            "customer_info": customer_info,
            "order_history": order_history,
            "order_summary": order_summary
        }

    def calculate_rfm_score(self, customer_id: str) -> Dict[str, float]:
        """Calculate RFM (Recency, Frequency, Monetary) score"""
        customer_orders = self.orders_df[self.orders_df['customer_id'] == customer_id]

        if customer_orders.empty:
            return {"recency": 0, "frequency": 0, "monetary": 0, "rfm_score": 0}

        # Calculate current date
        current_date = datetime.now()

        # Recency: Days since last purchase (lower is better)
        last_purchase = customer_orders['order_date'].max()
        recency_days = (current_date - last_purchase).days

        # Frequency: Number of orders
        frequency = len(customer_orders)

        # Monetary: Total spent
        monetary = customer_orders['total'].sum()

        # Normalize scores (simple linear scaling)
        # For recency: 0-30 days = 100, >90 days = 0
        recency_score = max(0, min(100, 100 - (recency_days * 100 / 90)))

        # For frequency: 1 order = 20, 5+ orders = 100
        frequency_score = min(100, frequency * 20)

        # For monetary: Scale based on data range
        max_monetary = self.orders_df.groupby('customer_id')['total'].sum().max()
        monetary_score = min(100, (monetary / max_monetary) * 100) if max_monetary > 0 else 0

        # Combined RFM score (weighted average)
        rfm_score = (recency_score * 0.3 + frequency_score * 0.3 + monetary_score * 0.4)

        return {
            "recency": recency_score,
            "frequency": frequency_score,
            "monetary": monetary_score,
            "rfm_score": int(rfm_score)
        }

    def calculate_churn_risk(self, customer_id: str) -> float:
        """Calculate churn risk score (0-1)"""
        customer_data = self.get_customer_data(customer_id)

        if not customer_data or customer_data["order_summary"]["total_orders"] == 0:
            return 1.0  # High churn risk for no orders

        order_summary = customer_data["order_summary"]
        days_since_last_order = order_summary["days_since_last_order"]
        total_orders = order_summary["total_orders"]
        avg_order_value = order_summary["avg_order_value"]

        # Calculate churn risk factors
        recency_risk = min(1.0, days_since_last_order / 60)  # Risk increases after 60 days
        frequency_risk = 1.0 / (1 + total_orders)  # More orders = lower risk
        value_risk = 1.0 / (1 + avg_order_value / 10)  # Higher AOV = lower risk

        # Weighted churn risk
        churn_risk = (recency_risk * 0.5 + frequency_risk * 0.3 + value_risk * 0.2)

        return round(min(1.0, churn_risk), 3)

    def get_top_followups_for_date(self, target_date: date, limit: int = 5) -> List[str]:
        """Get top customers to follow up on a specific date"""
        if self.customers_df is None:
            return []

        customer_priorities = []

        for customer_id in self.customers_df['customer_id']:
            rfm_data = self.calculate_rfm_score(customer_id)
            churn_risk = self.calculate_churn_risk(customer_id)

            # Priority score based on RFM and churn risk
            priority_score = (rfm_data["rfm_score"] * 0.6) + ((1 - churn_risk) * 40)

            customer_priorities.append({
                "customer_id": customer_id,
                "priority_score": priority_score,
                "churn_risk": churn_risk
            })

        # Sort by priority score (descending) and churn risk (ascending)
        customer_priorities.sort(key=lambda x: (-x["priority_score"], x["churn_risk"]))

        return [cp["customer_id"] for cp in customer_priorities[:limit]]
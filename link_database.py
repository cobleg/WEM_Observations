import sqlite3

def generate_semantic_graph_from_wem_db(db_path, service_type='energy'):
    """Generate semantic statements from WEM database for a specific service."""
    conn = sqlite3.connect(db_path)
    
    # Load price data
    price_query = f"""
    SELECT interval_time, price 
    FROM dispatch_prices
    WHERE market_service = '{service_type}'
    ORDER BY interval_time
    """
    price_df = pd.read_sql_query(price_query, conn, parse_dates=['interval_time'], index_col='interval_time')
    
    # Detect price events
    price_events = detect_events(price_df, 'price', threshold=2.5)
    
    # Create semantic graph
    g = create_semantic_statements(price_events, service_type)
    
    # For facilities, we can also analyze their dispatch quantities
    if service_type == 'energy':
        facilities_query = """
        SELECT DISTINCT facility_name
        FROM dispatch_quantities
        WHERE market_service = 'energy'
        """
        facilities = pd.read_sql_query(facilities_query, conn)['facility_name'].tolist()
        
        for facility in facilities:
            facility_query = f"""
            SELECT interval_time, quantity 
            FROM dispatch_quantities
            WHERE market_service = '{service_type}' AND facility_name = '{facility}'
            ORDER BY interval_time
            """
            facility_df = pd.read_sql_query(facility_query, conn, parse_dates=['interval_time'], index_col='interval_time')
            
            # Detect quantity events
            quantity_events = detect_events(facility_df, 'quantity', threshold=2.5)
            
            # Add to semantic graph
            facility_g = create_semantic_statements(quantity_events, service_type, facility)
            g += facility_g
    
    conn.close()
    return g

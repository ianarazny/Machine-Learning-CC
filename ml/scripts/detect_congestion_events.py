def detect_congestion_events(df, loss_threshold=1, retrans_threshold=0, 
                           cwnd_drop_pct=0.1, rtt_spike_factor=1.2):
    
    # Señales directas de congestión
    packet_loss = (df['packets_lost'].diff() > loss_threshold)
    retransmissions = (df['bytes_retrans'].diff() > retrans_threshold)
    
    # Señales indirectas
    cwnd_significant_drop = (df['cwnd'].pct_change() < -cwnd_drop_pct)
    rtt_spike = (df['rtt'] > df['rtt'].rolling(5).mean() * rtt_spike_factor)
    throughput_drop = (df['throughput'].pct_change() < -0.15)  # 15% drop
    
    # Combinar señales (OR lógico con pesos)
    congestion_event = (
        packet_loss |           # Señal más directa
        retransmissions |       # Muy confiable
        (cwnd_significant_drop & rtt_spike) |  # Combinación fuerte
        (throughput_drop & rtt_spike)          # Degradación observable
    ).astype(int)
    
    return congestion_event
-- Users
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW()
);

-- User Settings (one per user)
CREATE TABLE user_settings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    location VARCHAR(255) DEFAULT 'Delhi,IN',
    battery_capacity_kwh FLOAT DEFAULT 15.0,
    solar_capacity_kw FLOAT DEFAULT 100.0,
    mode VARCHAR(50) DEFAULT 'tou_savings',
    -- modes: self_consumption | tou_savings | full_backup | low_power
    appliances JSONB DEFAULT '["washing_machine","dishwasher","pump"]',
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Cached Weather Data (avoid hitting API repeatedly)
CREATE TABLE weather_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    location VARCHAR(255) NOT NULL,
    fetched_at TIMESTAMP DEFAULT NOW(),
    data JSONB NOT NULL  -- stores all 24 hourly readings
);

-- Historical Tariff Data
CREATE TABLE tariff_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    recorded_at TIMESTAMP NOT NULL,
    hour INT NOT NULL,        -- 0-23
    tariff_value FLOAT NOT NULL,
    location VARCHAR(255)
);

-- Optimization Results (for Analytics page history)
CREATE TABLE optimization_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    created_at TIMESTAMP DEFAULT NOW(),
    mode VARCHAR(50),
    total_grid_cost FLOAT,
    recommendations JSONB,
    hourly_schedule JSONB,    -- full 24hr schedule
    estimated_savings FLOAT
);

-- Notifications
CREATE TABLE notifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    created_at TIMESTAMP DEFAULT NOW(),
    message TEXT NOT NULL,
    type VARCHAR(50),  -- suggestion | alert | info
    is_read BOOLEAN DEFAULT FALSE
);
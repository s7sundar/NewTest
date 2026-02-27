import { Sequelize, DataTypes } from 'sequelize';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const sequelize = new Sequelize({
  dialect: 'sqlite',
  storage: path.join(__dirname, '../../revulog_demo.db'),
  logging: false
});

const Location = sequelize.define('Location', {
  id: {
    type: DataTypes.INTEGER,
    primaryKey: true,
    autoIncrement: true
  },
  name: {
    type: DataTypes.STRING,
    allowNull: false
  },
  business_type: {
    type: DataTypes.STRING,
    allowNull: false
  },
  address: {
    type: DataTypes.STRING
  },
  engine_tier: {
    type: DataTypes.STRING,
    defaultValue: 'MICRO'
  },
  monthly_volume: {
    type: DataTypes.FLOAT,
    defaultValue: 0
  },
  overall_rating: {
    type: DataTypes.FLOAT,
    defaultValue: 4.0
  },
  total_reviews: {
    type: DataTypes.INTEGER,
    defaultValue: 0
  },
  created_at: {
    type: DataTypes.DATE,
    defaultValue: DataTypes.NOW
  }
}, {
  tableName: 'locations',
  timestamps: false
});

const Review = sequelize.define('Review', {
  id: {
    type: DataTypes.INTEGER,
    primaryKey: true,
    autoIncrement: true
  },
  location_id: {
    type: DataTypes.INTEGER,
    allowNull: false
  },
  reviewer_name: {
    type: DataTypes.STRING
  },
  rating: {
    type: DataTypes.INTEGER,
    allowNull: false
  },
  text: {
    type: DataTypes.TEXT
  },
  date: {
    type: DataTypes.DATEONLY,
    allowNull: false
  },
  focus_areas: {
    type: DataTypes.STRING
  },
  issue_weight: {
    type: DataTypes.FLOAT,
    defaultValue: 0
  },
  has_escalation: {
    type: DataTypes.BOOLEAN,
    defaultValue: false
  },
  created_at: {
    type: DataTypes.DATE,
    defaultValue: DataTypes.NOW
  }
}, {
  tableName: 'reviews',
  timestamps: false
});

const DailyAggregate = sequelize.define('DailyAggregate', {
  id: {
    type: DataTypes.INTEGER,
    primaryKey: true,
    autoIncrement: true
  },
  location_id: {
    type: DataTypes.INTEGER,
    allowNull: false
  },
  focus_area: {
    type: DataTypes.STRING,
    allowNull: false
  },
  date: {
    type: DataTypes.DATEONLY,
    allowNull: false
  },
  total_reviews: {
    type: DataTypes.INTEGER,
    defaultValue: 0
  },
  issue_count: {
    type: DataTypes.INTEGER,
    defaultValue: 0
  },
  issue_rate: {
    type: DataTypes.FLOAT,
    defaultValue: 0
  },
  negative_count: {
    type: DataTypes.INTEGER,
    defaultValue: 0
  }
}, {
  tableName: 'daily_aggregates',
  timestamps: false
});

const FocusAreaState = sequelize.define('FocusAreaState', {
  id: {
    type: DataTypes.INTEGER,
    primaryKey: true,
    autoIncrement: true
  },
  location_id: {
    type: DataTypes.INTEGER,
    allowNull: false
  },
  focus_area: {
    type: DataTypes.STRING,
    allowNull: false
  },
  engine_tier: {
    type: DataTypes.STRING,
    allowNull: false
  },
  current_state: {
    type: DataTypes.STRING,
    allowNull: false
  },
  previous_state: {
    type: DataTypes.STRING
  },
  confidence: {
    type: DataTypes.STRING,
    defaultValue: 'Medium'
  },
  confidence_score: {
    type: DataTypes.FLOAT,
    defaultValue: 0
  },
  is_recurring: {
    type: DataTypes.BOOLEAN,
    defaultValue: false
  },
  period_days: {
    type: DataTypes.FLOAT
  },
  peak_lift: {
    type: DataTypes.FLOAT
  },
  stability: {
    type: DataTypes.FLOAT
  },
  spike_count: {
    type: DataTypes.INTEGER,
    defaultValue: 0
  },
  trend_direction: {
    type: DataTypes.STRING
  },
  evidence_summary: {
    type: DataTypes.TEXT
  },
  last_updated: {
    type: DataTypes.DATE,
    defaultValue: DataTypes.NOW
  }
}, {
  tableName: 'focus_area_states',
  timestamps: false
});

Location.hasMany(Review, { foreignKey: 'location_id' });
Review.belongsTo(Location, { foreignKey: 'location_id' });

Location.hasMany(DailyAggregate, { foreignKey: 'location_id' });
DailyAggregate.belongsTo(Location, { foreignKey: 'location_id' });

Location.hasMany(FocusAreaState, { foreignKey: 'location_id' });
FocusAreaState.belongsTo(Location, { foreignKey: 'location_id' });

export { sequelize, Location, Review, DailyAggregate, FocusAreaState };

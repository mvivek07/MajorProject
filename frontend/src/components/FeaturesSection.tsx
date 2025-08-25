
import React from 'react';
import { 
  Activity, 
  AlertTriangle, 
  TrendingDown, 
  SlidersHorizontal, 
  Upload, 
  BarChart2 
} from "lucide-react";

const FeaturesSection: React.FC = () => {
  const features = [
    {
      icon: <Activity className="h-8 w-8 text-cashwise-blue" />,
      title: "AI-Powered Predictions",
      description: "Machine learning algorithms that learn from your financial patterns to provide accurate cash flow forecasts."
    },
    {
      icon: <AlertTriangle className="h-8 w-8 text-cashwise-blue" />,
      title: "Risk Alerts",
      description: "Get notified of potential cash shortages or financial issues before they become critical problems."
    },
    {
      icon: <TrendingDown className="h-8 w-8 text-cashwise-blue" />,
      title: "Cash Burn Analysis",
      description: "Understand your burn rate and runway with visual analytics and actionable recommendations."
    },
    {
      icon: <SlidersHorizontal className="h-8 w-8 text-cashwise-blue" />,
      title: "Budget Optimizer",
      description: "AI-driven suggestions for optimizing expenses and improving your overall financial health."
    },
    {
      icon: <Upload className="h-8 w-8 text-cashwise-blue" />,
      title: "Easy CSV/Excel Upload",
      description: "Simple data import options that work with your existing financial tracking systems."
    },
    {
      icon: <BarChart2 className="h-8 w-8 text-cashwise-blue" />,
      title: "Financial Health Score",
      description: "Get a holistic view of your financial well-being with our proprietary scoring system."
    }
  ];

  return (
    <section className="section-padding bg-blue-50">
      <div className="container mx-auto px-4">
        <div className="text-center max-w-3xl mx-auto mb-16">
          <h2 className="text-3xl md:text-4xl font-bold">Powerful Features</h2>
          <p className="mt-4 text-lg text-gray-600">
            Our comprehensive suite of tools gives you everything you need to master your cash flow management.
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <div 
              key={index} 
              className="feature-card flex items-start animate-fade-in"
              style={{ animationDelay: `${index * 100}ms` }}
            >
              <div className="mr-4">
                <div className="h-12 w-12 rounded-full bg-cashwise-blue-light/30 flex items-center justify-center">
                  {feature.icon}
                </div>
              </div>
              <div>
                <h3 className="text-lg font-bold mb-2">{feature.title}</h3>
                <p className="text-gray-600 text-sm">{feature.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default FeaturesSection;

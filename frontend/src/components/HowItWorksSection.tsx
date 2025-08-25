
import React from 'react';
import { FileText, LineChart, Lightbulb } from "lucide-react";

const HowItWorksSection: React.FC = () => {
  const steps = [
    {
      icon: <FileText className="h-10 w-10 text-cashwise-blue" />,
      title: "Connect your Data",
      description: "Easily connect your bank feeds, accounting software, or upload financial data directly via CSV/Excel."
    },
    {
      icon: <LineChart className="h-10 w-10 text-cashwise-blue" />,
      title: "ML Model Prediction",
      description: "Our advanced algorithms use Random Forest and time-series models to predict future cash flows with high accuracy."
    },
    {
      icon: <Lightbulb className="h-10 w-10 text-cashwise-blue" />,
      title: "Actionable Insights",
      description: "Receive alerts, budgeting suggestions, and optimization recommendations based on your unique financial patterns."
    },
  ];

  return (
    <section id="how-it-works" className="section-padding bg-gradient-to-b from-blue-50 to-white">
      <div className="container mx-auto px-4">
        <div className="text-center max-w-3xl mx-auto mb-16">
          <h2 className="text-3xl md:text-4xl font-bold">How CashWise AI Works</h2>
          <p className="mt-4 text-lg text-gray-600">
            Our platform uses sophisticated machine learning to transform your financial data into accurate predictions and actionable insights.
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-5xl mx-auto">
          {steps.map((step, index) => (
            <div 
              key={index}
              className="feature-card flex flex-col items-center text-center animate-fade-in"
              style={{ animationDelay: `${index * 200}ms` }}
            >
              <div className="h-20 w-20 rounded-full bg-cashwise-blue-light/30 flex items-center justify-center mb-6">
                {step.icon}
              </div>
              <h3 className="text-xl font-bold mb-3">{step.title}</h3>
              <p className="text-gray-600">{step.description}</p>
            </div>
          ))}
        </div>
        
        <div className="mt-16 text-center">
          <div className="inline-block p-2 px-4 rounded-full bg-cashwise-green-light text-sm font-medium text-gray-700">
            Trusted by 1,000+ businesses and individuals
          </div>
        </div>
      </div>
    </section>
  );
};

export default HowItWorksSection;

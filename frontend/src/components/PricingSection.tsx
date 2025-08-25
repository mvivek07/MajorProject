
import React from 'react';
import { Button } from "@/components/ui/button";
import { Check } from "lucide-react";

const PricingSection: React.FC = () => {
  const plans = [
    {
      name: "Individual",
      description: "Perfect for freelancers and personal finances",
      price: "Free",
      frequency: "",
      features: [
        "Basic cash flow predictions",
        "CSV/Excel uploads",
        "1 financial account",
        "30-day forecast horizon",
        "Email alerts"
      ],
      cta: "Get Started",
      highlighted: false
    },
    {
      name: "SMB",
      description: "For growing small and medium businesses",
      price: "$49",
      frequency: "/month",
      features: [
        "Advanced AI predictions",
        "Accounting software integration",
        "Unlimited financial accounts",
        "90-day forecast horizon",
        "Scenario planning",
        "Weekly detailed reports",
        "Priority email support"
      ],
      cta: "Start 14-Day Trial",
      highlighted: true
    },
    {
      name: "Enterprise",
      description: "For larger organizations with complex needs",
      price: "Custom",
      frequency: "",
      features: [
        "All SMB features",
        "Custom ML model training",
        "API access",
        "180-day forecast horizon",
        "Multi-company support",
        "Advanced financial metrics",
        "Dedicated account manager"
      ],
      cta: "Contact Sales",
      highlighted: false
    }
  ];

  return (
    <section id="pricing" className="section-padding bg-gradient-to-b from-blue-50 to-white">
      <div className="container mx-auto px-4">
        <div className="text-center max-w-3xl mx-auto mb-16">
          <h2 className="text-3xl md:text-4xl font-bold">Simple, Transparent Pricing</h2>
          <p className="mt-4 text-lg text-gray-600">
            Choose the plan that fits your needs. All plans include our core AI-powered forecasting technology.
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-5xl mx-auto">
          {plans.map((plan, index) => (
            <div 
              key={index} 
              className={`pricing-card flex flex-col animate-fade-in ${
                plan.highlighted ? 'border-cashwise-blue ring-2 ring-cashwise-blue-light' : ''
              }`}
              style={{ animationDelay: `${index * 150}ms` }}
            >
              {plan.highlighted && (
                <div className="absolute -top-4 left-0 right-0 flex justify-center">
                  <span className="bg-cashwise-blue text-white text-xs font-semibold px-3 py-1 rounded-full">
                    Most Popular
                  </span>
                </div>
              )}
              
              <div className="mb-6">
                <h3 className="text-xl font-bold">{plan.name}</h3>
                <p className="text-gray-600 text-sm mt-1">{plan.description}</p>
              </div>
              
              <div className="mb-6">
                <div className="flex items-baseline">
                  <span className="text-4xl font-bold">{plan.price}</span>
                  <span className="text-gray-600 ml-1">{plan.frequency}</span>
                </div>
              </div>
              
              <ul className="space-y-3 mb-8">
                {plan.features.map((feature, i) => (
                  <li key={i} className="flex items-start text-sm">
                    <Check className="h-4 w-4 text-cashwise-blue mr-2 mt-0.5 shrink-0" />
                    <span>{feature}</span>
                  </li>
                ))}
              </ul>
              
              <div className="mt-auto">
                <Button 
                  className={`w-full ${
                    plan.highlighted ? 'bg-cashwise-blue hover:bg-cashwise-blue-dark' : ''
                  }`}
                  variant={plan.highlighted ? "default" : "outline"}
                >
                  {plan.cta}
                </Button>
              </div>
            </div>
          ))}
        </div>
        
        <div className="text-center mt-12 text-sm text-gray-500 max-w-xl mx-auto">
          All plans include our secure data encryption, regular feature updates, and our satisfaction guarantee. Need a custom solution? Contact our sales team for tailored enterprise options.
        </div>
      </div>
    </section>
  );
};

export default PricingSection;

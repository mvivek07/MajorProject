
import React from 'react';
import { Button } from "@/components/ui/button";
import { ArrowRight, PlayCircle } from "lucide-react";

const HeroSection: React.FC = () => {
  return (
    <section className="pt-32 pb-16 md:pt-40 md:pb-24 overflow-hidden bg-gradient-to-b from-white to-blue-50">
      <div className="container mx-auto px-4">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          <div className="animate-fade-in [animation-delay:200ms]">
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold tracking-tight">
              AI-Driven Cash Flow <span className="gradient-text">Clarity</span> for Your Business and Life
            </h1>
            <p className="mt-6 text-lg md:text-xl text-gray-600 max-w-xl">
              Predict, Manage, and Optimize Cash Flow using Machine Learning. Get ahead of financial surprises with our powerful AI forecasting tools.
            </p>
            <div className="mt-8 flex flex-wrap gap-4">
              <Button size="lg" className="group">
                Try CashWise Free
                <ArrowRight className="ml-2 h-4 w-4 group-hover:translate-x-1 transition" />
              </Button>
              <Button size="lg" variant="outline" className="flex items-center">
                <PlayCircle className="mr-2 h-5 w-5" />
                See How It Works
              </Button>
            </div>
          </div>
          
          <div className="relative animate-fade-in [animation-delay:400ms]">
            <div className="absolute -top-28 -right-28 w-96 h-96 bg-cashwise-blue/20 rounded-full blur-3xl"></div>
            <div className="absolute -bottom-28 -left-28 w-96 h-96 bg-cashwise-green/20 rounded-full blur-3xl"></div>
            
            <div className="relative bg-white rounded-xl shadow-lg p-4 border">
              <div className="bg-gray-50 rounded-lg h-64 overflow-hidden flex items-center justify-center">
                <div className="w-full h-full bg-gradient-to-br from-cashwise-blue-light to-cashwise-green-light p-4">
                  <div className="bg-white rounded-lg h-full w-full flex items-center justify-center">
                    <div className="w-5/6 h-4/5">
                      <div className="w-full h-full flex flex-col">
                        <div className="flex justify-between items-center mb-4">
                          <div className="text-sm font-bold">Cash Flow Forecast</div>
                          <div className="text-xs text-gray-500">Last 6 months</div>
                        </div>
                        <div className="relative flex-grow">
                          {/* Simple chart visualization */}
                          <div className="absolute bottom-0 left-0 right-0 h-3/5 flex items-end space-x-1">
                            {[40, 65, 35, 50, 70, 55, 45, 60, 75, 50, 80, 65].map((value, index) => (
                              <div 
                                key={index} 
                                className="flex-1 bg-gradient-to-t from-cashwise-blue to-cashwise-blue-light rounded-t"
                                style={{ height: `${value}%`, animationDelay: `${index * 100}ms` }}
                              ></div>
                            ))}
                          </div>
                        </div>
                        <div className="h-px bg-gray-200 my-2"></div>
                        <div className="flex justify-between text-xs text-gray-500">
                          <div>Jan</div>
                          <div>Mar</div>
                          <div>Jun</div>
                          <div>Sep</div>
                          <div>Dec</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              <div className="mt-4 p-2 bg-gray-50 rounded-lg">
                <div className="flex justify-between items-center">
                  <div className="text-sm font-medium">Predicted Cash Position</div>
                  <div className="text-lg font-bold text-cashwise-blue">$124,500</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default HeroSection;

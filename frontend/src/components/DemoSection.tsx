
import React, { useState } from 'react';
import { Button } from "@/components/ui/button";
import { Upload, RefreshCw } from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const DemoSection: React.FC = () => {
  const [isLoading, setIsLoading] = useState(false);
  
  const handleUpload = () => {
    setIsLoading(true);
    setTimeout(() => {
      setIsLoading(false);
    }, 2000);
  };
  
  return (
    <section className="section-padding">
      <div className="container mx-auto px-4">
        <div className="text-center max-w-3xl mx-auto mb-12">
          <h2 className="text-3xl md:text-4xl font-bold">See CashWise AI in Action</h2>
          <p className="mt-4 text-lg text-gray-600">
            Try our interactive demo to see how our AI predicts cash flow patterns or upload your own data for a personalized forecast.
          </p>
        </div>
        
        <div className="max-w-4xl mx-auto">
          <Tabs defaultValue="demo" className="w-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="demo">Interactive Demo</TabsTrigger>
              <TabsTrigger value="upload">Upload Your Data</TabsTrigger>
            </TabsList>
            
            <TabsContent value="demo" className="mt-6">
              <Card>
                <CardHeader>
                  <CardTitle>Cash Flow Prediction Demo</CardTitle>
                  <CardDescription>Based on sample data from a retail business</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="aspect-video bg-gray-50 rounded-lg overflow-hidden">
                    <div className="w-full h-full p-4">
                      <div className="mb-4 flex justify-between items-center">
                        <div className="text-sm font-medium">Projected Cash Flow</div>
                        <div className="flex gap-2">
                          <Button variant="outline" size="sm">Last 6 Months</Button>
                          <Button variant="outline" size="sm">Next 6 Months</Button>
                        </div>
                      </div>
                      
                      <div className="h-72 bg-white rounded-lg border p-4">
                        <div className="h-full flex flex-col">
                          <div className="flex-1 relative">
                            {/* Chart area */}
                            <div className="absolute inset-0">
                              {/* Line representing cash flow */}
                              <div className="absolute bottom-1/2 left-0 right-0 h-px bg-gray-200"></div>
                              
                              {/* Cash flow line */}
                              <svg className="absolute inset-0" viewBox="0 0 100 50" preserveAspectRatio="none">
                                {/* Historical data - solid line */}
                                <path 
                                  d="M0,30 L10,25 L20,35 L30,20 L40,30 L50,25" 
                                  fill="none" 
                                  stroke="#33C3F0" 
                                  strokeWidth="0.5"
                                />
                                
                                {/* Projected data - dashed line */}
                                <path 
                                  d="M50,25 L60,15 L70,25 L80,10 L90,20 L100,15" 
                                  fill="none" 
                                  stroke="#33C3F0" 
                                  strokeWidth="0.5" 
                                  strokeDasharray="1"
                                />
                                
                                {/* Area under the curve */}
                                <path 
                                  d="M0,30 L10,25 L20,35 L30,20 L40,30 L50,25 L60,15 L70,25 L80,10 L90,20 L100,15 L100,50 L0,50 Z" 
                                  fill="url(#blue-gradient)" 
                                  opacity="0.2"
                                />
                                
                                <defs>
                                  <linearGradient id="blue-gradient" x1="0%" y1="0%" x2="0%" y2="100%">
                                    <stop offset="0%" stopColor="#33C3F0" stopOpacity="1" />
                                    <stop offset="100%" stopColor="#33C3F0" stopOpacity="0" />
                                  </linearGradient>
                                </defs>
                              </svg>
                              
                              {/* Vertical now line */}
                              <div className="absolute top-0 bottom-0 left-1/2 w-px bg-gray-300 before:absolute before:top-0 before:-left-1 before:h-2 before:w-2 before:rounded-full before:bg-gray-300"></div>
                            </div>
                          </div>
                          
                          <div className="h-8 flex justify-between text-xs text-gray-500 border-t pt-2">
                            <div>Jan</div>
                            <div>Feb</div>
                            <div>Mar</div>
                            <div>Apr</div>
                            <div>May</div>
                            <div>Jun</div>
                            <div>Jul</div>
                            <div>Aug</div>
                            <div>Sep</div>
                            <div>Oct</div>
                            <div>Nov</div>
                            <div>Dec</div>
                          </div>
                        </div>
                      </div>
                      
                      <div className="mt-4 grid grid-cols-3 gap-4 text-center">
                        <div className="p-3 bg-white rounded-lg border">
                          <p className="text-xs text-gray-500">Cash Burn Rate</p>
                          <p className="text-xl font-bold text-cashwise-blue">$12,450/mo</p>
                        </div>
                        <div className="p-3 bg-white rounded-lg border">
                          <p className="text-xs text-gray-500">Runway</p>
                          <p className="text-xl font-bold text-cashwise-blue">8.2 months</p>
                        </div>
                        <div className="p-3 bg-white rounded-lg border">
                          <p className="text-xs text-gray-500">Health Score</p>
                          <p className="text-xl font-bold text-green-500">72/100</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
                <CardFooter>
                  <Button className="w-full">
                    Try With Your Data
                  </Button>
                </CardFooter>
              </Card>
            </TabsContent>
            
            <TabsContent value="upload" className="mt-6">
              <Card>
                <CardHeader>
                  <CardTitle>Upload Your Financial Data</CardTitle>
                  <CardDescription>
                    Securely upload CSV/Excel files to get personalized predictions
                  </CardDescription>
                </CardHeader>
                <CardContent className="flex flex-col items-center justify-center py-12">
                  <div className="w-full max-w-md border-2 border-dashed border-gray-200 rounded-lg p-12 flex flex-col items-center justify-center">
                    <Upload className="h-12 w-12 text-gray-300 mb-4" />
                    <p className="text-lg font-medium text-center mb-2">
                      Drag and drop your CSV or Excel file
                    </p>
                    <p className="text-sm text-gray-500 text-center mb-6">
                      Or click to browse your files
                    </p>
                    <Button onClick={handleUpload}>
                      {isLoading ? (
                        <>
                          <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                          Processing...
                        </>
                      ) : (
                        "Select File"
                      )}
                    </Button>
                  </div>
                </CardContent>
                <CardFooter className="flex flex-col space-y-2 items-start">
                  <p className="text-xs text-gray-500">
                    Supported formats: .csv, .xlsx (up to 5MB)
                  </p>
                  <p className="text-xs text-gray-500">
                    Your data is processed securely and never shared with third parties.
                  </p>
                </CardFooter>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </section>
  );
};

export default DemoSection;

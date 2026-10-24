#import <Foundation/Foundation.h>
#include "engine.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

static int32_t integer(id value) {
    if (![value isKindOfClass:NSNumber.class]) throw std::runtime_error("Expected integer token/mask");
    double n=[value doubleValue];
    if(n<INT32_MIN || n>INT32_MAX || std::floor(n)!=n) throw std::runtime_error("Token/mask exceeds int32");
    return static_cast<int32_t>(n);
}

int main(int argc,char** argv) {
    @autoreleasepool {
        try {
            std::string path,dtype="fp32",matmul="auto"; size_t dim=768;
            for(int i=1;i<argc;++i) {
                std::string arg=argv[i];
                if(arg=="--info") { std::cout<<egemma::device_info()<<'\n'; return 0; }
                if(arg=="--help") {
                    std::cout<<"egemma-cli --model-dir PATH [--dtype fp32|fp16] [--dim 768|512|256|128] [--matmul auto|native|mpp]\n"
                        <<"stdin JSONL: token array or {input_ids: [...], attention_mask: [...]}\n"
                        <<"input_ids may also contain a batch of variable-length arrays.\n"
                        <<"stdout JSONL: one vector array per input sequence.\n";
                    return 0;
                }
                if(i+1>=argc) throw std::runtime_error("Missing value for " + arg);
                if(arg=="--model-dir") path=argv[++i];
                else if(arg=="--dtype") dtype=argv[++i];
                else if(arg=="--dim") dim=std::stoul(argv[++i]);
                else if(arg=="--matmul") matmul=argv[++i];
                else throw std::runtime_error("Unknown option " + arg);
            }
            if(path.empty()) throw std::runtime_error("--model-dir is required");
            egemma::Engine engine(path,dtype,false,matmul);
            std::string line; size_t line_number=0;
            while(std::getline(std::cin,line)) @autoreleasepool {
                ++line_number; if(line.empty()) continue;
                NSData* data=[NSData dataWithBytes:line.data() length:line.size()];
                NSError* error=nil;
                id value=[NSJSONSerialization JSONObjectWithData:data options:NSJSONReadingFragmentsAllowed error:&error];
                if(!value) throw std::runtime_error("Invalid JSON on line " + std::to_string(line_number));
                NSArray* rows=[value isKindOfClass:NSDictionary.class] ? value[@"input_ids"] : value;
                NSArray* masks=[value isKindOfClass:NSDictionary.class] ? value[@"attention_mask"] : nil;
                if(![rows isKindOfClass:NSArray.class] || rows.count==0) throw std::runtime_error("Nonempty input_ids array required");
                if(![rows[0] isKindOfClass:NSArray.class]) { rows=@[rows]; if(masks) masks=@[masks]; }
                if(masks && (![masks isKindOfClass:NSArray.class] || masks.count!=rows.count)) throw std::runtime_error("Mask batch size mismatch");
                size_t B=rows.count,T=0;
                for(id row in rows) {
                    if(![row isKindOfClass:NSArray.class] || [row count]==0) throw std::runtime_error("Nonempty token rows required");
                    T=std::max(T,size_t([row count]));
                }
                if(B>SIZE_MAX/T) throw std::runtime_error("Input size overflow");
                std::vector<int32_t> ids(B*T,0),mask(B*T,0);
                for(size_t b=0;b<B;++b) {
                    NSArray* row=rows[b]; NSArray* mr=masks ? masks[b] : nil;
                    if(mr && (![mr isKindOfClass:NSArray.class] || mr.count!=row.count)) throw std::runtime_error("Mask length mismatch");
                    for(size_t t=0;t<row.count;++t) { ids[b*T+t]=integer(row[t]); mask[b*T+t]=mr ? integer(mr[t]) : 1; }
                }
                auto out=engine.encode(ids.data(),mask.data(),B,T,dim);
                for(size_t b=0;b<B;++b) {
                    NSMutableArray* vector=[NSMutableArray arrayWithCapacity:dim];
                    for(size_t j=0;j<dim;++j) [vector addObject:@(out[b*dim+j])];
                    NSData* result=[NSJSONSerialization dataWithJSONObject:vector options:0 error:&error];
                    if(!result) throw std::runtime_error("Cannot serialize output");
                    std::cout.write(static_cast<const char*>(result.bytes),result.length); std::cout<<'\n';
                }
                std::cout.flush();
            }
            return 0;
        } catch(const std::exception& error) { std::cerr<<"egemma-cli: "<<error.what()<<'\n'; return 1; }
    }
}

output = read.table("outputs/out.called.0.98")
target = read.table("basedata/target.txt")

if(all(output == target)) {
    print("Test succeeded")
}else{
    print("Test failed")
}


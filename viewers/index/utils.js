

function mean(arr) {
            let sum = 0;
            for(let i = 0; i < arr.length; i++) sum += arr[i];
            return sum / arr.length;
        }

function std(arr) {
    const avg = mean(arr)
    let sum = 0;
    for(let i = 0; i < arr.length; i++) sum += Math.pow(arr[i] - avg, 2);
    return Math.sqrt(sum / arr.length);
}

function parseCsv(txt) {
    let csvLines = txt.split('\n');
    let keys = csvLines[0].split(',');
    let data = {};
    for(let k = 0; k < keys.length; k++) data[keys[k]] = [];
    let values;
    for(let l = 1; l < csvLines.length; l++) {
        values = csvLines[l].split(',');
        for(let v = 0; v < values.length; v++)
            if(isNaN(values[v]))
                data[keys[v]].push(values[v]);
            else
                data[keys[v]].push(Number(values[v]));

    }
    return data;
}

function sulci_side_list(keys) {
    let ss_list = [];
    for(let k = 0; k < keys.length; k++) {
        if(keys[k].substring(0, 4).localeCompare("ESI_") == 0)
            ss_list.push(keys[k].substring(4))
    }
    return ss_list.sort()
}


function dragoverHandler(evt) {
    evt.stopPropagation();
    evt.preventDefault();
    // Explicitly show this is a copy.
    evt.dataTransfer.dropEffect = 'copy';
}
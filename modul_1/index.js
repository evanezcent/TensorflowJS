import 'bootstrap/dist/css/bootstrap.css'
import $ from 'jquery'
import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import Chart from 'chart.js'

$('#output').text('Hello World')

const ar_data = [-1, 0, 1, 2, 3, 4]
const ar_target = [-3, -1, 1, 3, 5, 7]

const x = tf.tensor2d(ar_data, [6, 1])
const y = tf.tensor2d(ar_target, [6, 1])

let zip = (ar1, ar2) => ar1.map((x, i) => { return { 'x': x, 'y': ar2[i], } })

const data_train = zip(ar_data, ar_target)
const label_train = ['trainset'] //label tertampil pada chart

//TFJS-VIS (SHOW IN PAGE)
// let data = { values: [data_train], series: label_train }

// const container = $('#scatter-tfjs')[0]
// tfvis.render.scatterplot(container, data, { width:500, height:400 } )

//TFJS-VIS VISOR (SHOW IN POP UP)
// const surface = tfvis.visor().surface({ 
// 	name: 'Scatterplot-tfjs-visor', tab: 'Charts' })
// tfvis.render.scatterplot(surface, data);

// CHART.JS 
var ctx = $('#scatter-chartjs')
var scatterChart = new Chart(ctx, {
    type: 'scatter',
    data: {
        datasets: [{
            data: data_train,
            label: label_train,
            backgroundColor: 'red'
        }]
    },
    options: { responsive: false }
})

let model
$('#init-btn').click(function () {

    model = tf.sequential()
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }))
    model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' })

    let t_pred = model.predict(x)
    let y_pred = t_pred.dataSync()
    let data_pred = zip(ar_data, y_pred)

    viewPrediction(data_train, data_pred)
    $('#train-btn').prop('disabled', false)

})

function viewPrediction(scatterDt, lineDt) {
    scatterChart.destroy()
    scatterChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ar_data,
            datasets: [{
                type: 'line',
                label: 'prediction',
                data: lineDt,
                fill: false,
                borderColor: 'blue',
                pointRadius: 0
            }, {
                type: 'bubble',
                label: 'training data',
                data: scatterDt,
                backgroundColor: 'red',
                borderColor: 'transparent'
            }]
        },
        options: { responsive: false }
    })
}

$('#train-btn').click(function () {
    var msg = $('#is-training')
    msg.toggleClass('badge-warning')
    msg.text('Training, please wait...')

    model.fit(x, y, { epochs: 700 }).then((hist) => { //latih x dengan y sebanyak 20 kali
        let t_pred = model.predict(x)
        let y_pred = t_pred.dataSync()
        let data_pred = zip(ar_data, y_pred)
        viewPrediction(data_train, data_pred)

        let mse = model.evaluate(x, y);

        msg.removeClass('badge-warning').addClass('badge-success')
        msg.text('MSE: ' + mse.dataSync())  //seberapa jelek modelnya
        $('#predict').show()

        const surface = tfvis.visor().surface({
            name: 'Training History', tab: 'MSE'
        })
        tfvis.show.history(surface, hist, ['loss'])

    })
})

$('#predict-btn').click(function () {

    var num = parseFloat($('#inputValue').val())
    let y_pred = model.predict(tf.tensor2d([num], [1, 1]))

    $('#output').text(y_pred.dataSync())

})



